use std::collections::BTreeMap;
use std::env;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::VirtioShmRegion;
use crate::descriptor_utils;
use crate::descriptor_utils::{Reader, Writer};
use crate::protocol::GpuResponse::{
    ErrInvalidResourceId, ErrUnspec, OkCapset, OkCapsetInfo, OkMapInfo, OkNoData,
    OkResourcePlaneInfo, OkResourceUuid,
};
use crate::protocol::{
    virtio_gpu_ctrl_hdr, virtio_gpu_mem_entry, GpuCommand, GpuResponse, GpuResponsePlaneInfo,
    VirtioGpuResult, VIRTIO_GPU_BLOB_FLAG_CREATE_GUEST_HANDLE, VIRTIO_GPU_BLOB_MEM_HOST3D,
    VIRTIO_GPU_FLAG_FENCE,
};
use libc::c_void;
use rutabaga_gfx::{
    ResourceCreate3D, ResourceCreateBlob, Rutabaga, RutabagaBuilder, RutabagaChannel,
    RutabagaFence, RutabagaFenceHandler, RutabagaIovec, Transfer3D, RUTABAGA_CHANNEL_TYPE_WAYLAND,
    RUTABAGA_MAP_ACCESS_MASK, RUTABAGA_MAP_ACCESS_READ, RUTABAGA_MAP_ACCESS_RW,
    RUTABAGA_MAP_ACCESS_WRITE, RUTABAGA_MAP_CACHE_MASK, RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD,
    RUTABAGA_PIPE_BIND_RENDER_TARGET, RUTABAGA_PIPE_TEXTURE_2D,
};
use vhost::VhostAccess::No;
use vhost_user_backend::{VringRwLock, VringT};
use virtio_bindings::virtio_gpu::VIRTIO_GPU_FLAG_INFO_RING_IDX;
use virtio_bindings::virtio_mmio::VIRTIO_MMIO_INT_VRING;
use virtio_bindings::virtio_ring::vring;
use virtio_queue::{Queue, QueueT};
use vm_memory::{
    Bytes, GuestAddress, GuestAddressSpace, GuestMemory, GuestMemoryAtomic, GuestMemoryMmap,
    VolatileSlice,
};
use vmm_sys_util::eventfd::EventFd;

#[derive(Debug)]
// TODO: ThisErr
pub enum GpuError {
    /// Failed to create event fd.
    EventFd(std::io::Error),
    /// Failed to decode incoming command.
    DecodeCommand(std::io::Error),
    /// Error creating Reader for Queue.
    QueueReader(descriptor_utils::Error),
    /// Error creating Writer for Queue.
    QueueWriter(descriptor_utils::Error),
    /// Error writting to the Queue.
    WriteDescriptor(std::io::Error),
    /// Error reading Guest Memory,
    GuestMemory,
}

fn sglist_to_rutabaga_iovecs(
    vecs: &[(GuestAddress, usize)],
    mem: &GuestMemoryMmap,
) -> Result<Vec<RutabagaIovec>, GpuError> {
    if vecs
        .iter()
        .any(|&(addr, len)| mem.get_slice(addr, len).is_err())
    {
        return Err(GpuError::GuestMemory);
    }

    let mut rutabaga_iovecs: Vec<RutabagaIovec> = Vec::new();
    for &(addr, len) in vecs {
        let slice = mem.get_slice(addr, len).unwrap();
        rutabaga_iovecs.push(RutabagaIovec {
            base: slice.as_ptr() as *mut c_void,
            len,
        });
    }
    Ok(rutabaga_iovecs)
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum VirtioGpuRing {
    Global,
    ContextSpecific { ctx_id: u32, ring_idx: u8 },
}

struct FenceDescriptor {
    ring: VirtioGpuRing,
    fence_id: u64,
    desc_index: u16,
    len: u32,
}

#[derive(Default)]
pub struct FenceState {
    descs: Vec<FenceDescriptor>,
    completed_fences: BTreeMap<VirtioGpuRing, u64>,
}

struct VirtioGpuResource {
    size: u64,
    shmem_offset: Option<u64>,
    rutabaga_external_mapping: bool,
}

impl VirtioGpuResource {
    /// Creates a new VirtioGpuResource with the given metadata.  Width and height are used by the
    /// display, while size is useful for hypervisor mapping.
    pub fn new(_resource_id: u32, _width: u32, _height: u32, size: u64) -> VirtioGpuResource {
        VirtioGpuResource {
            size,
            shmem_offset: None,
            rutabaga_external_mapping: false,
        }
    }
}

pub struct VirtioGpu {
    rutabaga: Rutabaga,
    resources: BTreeMap<u32, VirtioGpuResource>,
    fence_state: Arc<Mutex<FenceState>>,
}

impl VirtioGpu {
    fn create_fence_handler(
        vring: VringRwLock,
        fence_state: Arc<Mutex<FenceState>>,
    ) -> RutabagaFenceHandler {
        RutabagaFenceHandler::new(move |completed_fence: RutabagaFence| {
            log::debug!(
                "XXX - fence called: id={}, ring_idx={}",
                completed_fence.fence_id,
                completed_fence.ring_idx
            );

            let mut fence_state = fence_state.lock().unwrap();
            let mut i = 0;

            let ring = match completed_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                0 => VirtioGpuRing::Global,
                _ => VirtioGpuRing::ContextSpecific {
                    ctx_id: completed_fence.ctx_id,
                    ring_idx: completed_fence.ring_idx,
                },
            };

            while i < fence_state.descs.len() {
                log::debug!("XXX - fence_id: {}", fence_state.descs[i].fence_id);
                if fence_state.descs[i].ring == ring
                    && fence_state.descs[i].fence_id <= completed_fence.fence_id
                {
                    let completed_desc = fence_state.descs.remove(i);
                    log::debug!(
                        "XXX - found fence: desc_index={}",
                        completed_desc.desc_index
                    );

                    if let Err(e) = vring.add_used(completed_desc.desc_index, completed_desc.len) {
                        log::error!(
                            "Failed to add_used descriptor inside Rutabaga fence handler: {e}"
                        );
                    }
                    if let Err(e) = vring.signal_used_queue() {
                        log::error!(
                            "Failed to signal used queue inside  Rutabaga fence handler: {e}"
                        );
                    }
                } else {
                    i += 1;
                }
            }
            // Update the last completed fence for this context
            fence_state
                .completed_fences
                .insert(ring, completed_fence.fence_id);
        })
    }

    pub fn new(vring: VringRwLock) -> Self {
        let xdg_runtime_dir = match env::var("XDG_RUNTIME_DIR") {
            Ok(dir) => dir,
            Err(_) => "/run/user/1000".to_string(),
        };
        let wayland_display = match env::var("WAYLAND_DISPLAY") {
            Ok(display) => display,
            Err(_) => "wayland-0".to_string(),
        };
        let path = PathBuf::from(format!("{}/{}", xdg_runtime_dir, wayland_display));

        let rutabaga_channels: Vec<RutabagaChannel> = vec![RutabagaChannel {
            base_channel: path,
            channel_type: RUTABAGA_CHANNEL_TYPE_WAYLAND,
        }];
        let rutabaga_channels_opt = Some(rutabaga_channels);

        let builder = RutabagaBuilder::new(rutabaga_gfx::RutabagaComponentType::VirglRenderer, 0)
            .set_rutabaga_channels(rutabaga_channels_opt)
            .set_use_egl(true)
            .set_use_gles(true)
            .set_use_glx(true)
            .set_use_surfaceless(true)
            .set_use_drm(true);

        let fence_state = Arc::new(Mutex::new(Default::default()));
        //TODO:
        let fence = Self::create_fence_handler(vring, fence_state.clone());
        let rutabaga = builder
            .build(fence, None)
            .expect("Rutabaga initialization failed!");

        Self {
            rutabaga,
            resources: Default::default(),
            fence_state,
        }
    }

    // Non-public function -- no doc comment needed!
    fn result_from_query(&mut self, resource_id: u32) -> GpuResponse {
        match self.rutabaga.query(resource_id) {
            Ok(query) => {
                let mut plane_info = Vec::with_capacity(4);
                for plane_index in 0..4 {
                    plane_info.push(GpuResponsePlaneInfo {
                        stride: query.strides[plane_index],
                        offset: query.offsets[plane_index],
                    });
                }
                let format_modifier = query.modifier;
                OkResourcePlaneInfo {
                    format_modifier,
                    plane_info,
                }
            }
            Err(_) => OkNoData,
        }
    }

    pub fn force_ctx_0(&self) {
        self.rutabaga.force_ctx_0()
    }

    /// Creates a 3D resource with the given properties and resource_id.
    pub fn resource_create_3d(
        &mut self,
        resource_id: u32,
        resource_create_3d: ResourceCreate3D,
    ) -> VirtioGpuResult {
        self.rutabaga
            .resource_create_3d(resource_id, resource_create_3d)?;

        let resource = VirtioGpuResource::new(
            resource_id,
            resource_create_3d.width,
            resource_create_3d.height,
            0,
        );

        // Rely on rutabaga to check for duplicate resource ids.
        self.resources.insert(resource_id, resource);
        Ok(self.result_from_query(resource_id))
    }

    /// Releases guest kernel reference on the resource.
    pub fn unref_resource(&mut self, resource_id: u32) -> VirtioGpuResult {
        let resource = self
            .resources
            .remove(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        if resource.rutabaga_external_mapping {
            self.rutabaga.unmap(resource_id)?;
        }

        self.rutabaga.unref_resource(resource_id)?;
        Ok(OkNoData)
    }

    /// If the resource is the scanout resource, flush it to the display.
    pub fn flush_resource(&mut self, resource_id: u32) -> VirtioGpuResult {
        if resource_id == 0 {
            return Ok(OkNoData);
        }

        #[cfg(windows)]
        match self.rutabaga.resource_flush(resource_id) {
            Ok(_) => return Ok(OkNoData),
            Err(RutabagaError::Unsupported) => {}
            Err(e) => return Err(ErrRutabaga(e)),
        }

        Ok(OkNoData)
    }

    /// Copies data to host resource from the attached iovecs. Can also be used to flush caches.
    pub fn transfer_write(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        transfer: Transfer3D,
    ) -> VirtioGpuResult {
        self.rutabaga
            .transfer_write(ctx_id, resource_id, transfer)?;
        Ok(OkNoData)
    }

    /// Copies data from the host resource to:
    ///    1) To the optional volatile slice
    ///    2) To the host resource's attached iovecs
    ///
    /// Can also be used to invalidate caches.
    pub fn transfer_read(
        &mut self,
        _ctx_id: u32,
        _resource_id: u32,
        _transfer: Transfer3D,
        _buf: Option<VolatileSlice>,
    ) -> VirtioGpuResult {
        panic!("virtio_gpu: transfer_read unimplemented");
    }

    /// Attaches backing memory to the given resource, represented by a `Vec` of `(address, size)`
    /// tuples in the guest's physical address space. Converts to RutabagaIovec from the memory
    /// mapping.
    pub fn attach_backing(
        &mut self,
        resource_id: u32,
        mem: &GuestMemoryMmap,
        vecs: Vec<(GuestAddress, usize)>,
    ) -> VirtioGpuResult {
        let rutabaga_iovecs = sglist_to_rutabaga_iovecs(&vecs[..], mem).map_err(|_| ErrUnspec)?;
        self.rutabaga.attach_backing(resource_id, rutabaga_iovecs)?;
        Ok(OkNoData)
    }

    /// Detaches any previously attached iovecs from the resource.
    pub fn detach_backing(&mut self, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga.detach_backing(resource_id)?;
        Ok(OkNoData)
    }

    /// Returns a uuid for the resource.
    pub fn resource_assign_uuid(&self, resource_id: u32) -> VirtioGpuResult {
        if !self.resources.contains_key(&resource_id) {
            return Err(ErrInvalidResourceId);
        }

        // TODO(stevensd): use real uuids once the virtio wayland protocol is updated to
        // handle more than 32 bits. For now, the virtwl driver knows that the uuid is
        // actually just the resource id.
        let mut uuid: [u8; 16] = [0; 16];
        for (idx, byte) in resource_id.to_be_bytes().iter().enumerate() {
            uuid[12 + idx] = *byte;
        }
        Ok(OkResourceUuid { uuid })
    }

    /// Gets rutabaga's capset information associated with `index`.
    pub fn get_capset_info(&self, index: u32) -> VirtioGpuResult {
        let (capset_id, version, size) = self.rutabaga.get_capset_info(index)?;
        Ok(OkCapsetInfo {
            capset_id,
            version,
            size,
        })
    }

    /// Gets a capset from rutabaga.
    pub fn get_capset(&self, capset_id: u32, version: u32) -> VirtioGpuResult {
        let capset = self.rutabaga.get_capset(capset_id, version)?;
        Ok(OkCapset(capset))
    }

    /// Creates a rutabaga context.
    pub fn create_context(
        &mut self,
        ctx_id: u32,
        context_init: u32,
        context_name: Option<&str>,
    ) -> VirtioGpuResult {
        self.rutabaga
            .create_context(ctx_id, context_init, context_name)?;
        Ok(OkNoData)
    }

    /// Destroys a rutabaga context.
    pub fn destroy_context(&mut self, ctx_id: u32) -> VirtioGpuResult {
        self.rutabaga.destroy_context(ctx_id)?;
        Ok(OkNoData)
    }

    /// Attaches a resource to a rutabaga context.
    pub fn context_attach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga.context_attach_resource(ctx_id, resource_id)?;
        Ok(OkNoData)
    }

    /// Detaches a resource from a rutabaga context.
    pub fn context_detach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga.context_detach_resource(ctx_id, resource_id)?;
        Ok(OkNoData)
    }

    /// Submits a command buffer to a rutabaga context.
    pub fn submit_command(
        &mut self,
        ctx_id: u32,
        commands: &mut [u8],
        fence_ids: &[u64],
    ) -> VirtioGpuResult {
        self.rutabaga.submit_command(ctx_id, commands, fence_ids)?;
        Ok(OkNoData)
    }

    /// Creates a fence with the RutabagaFence that can be used to determine when the previous
    /// command completed.
    pub fn create_fence(&mut self, rutabaga_fence: RutabagaFence) -> VirtioGpuResult {
        self.rutabaga.create_fence(rutabaga_fence)?;
        Ok(OkNoData)
    }

    pub fn process_fence(
        &mut self,
        ring: VirtioGpuRing,
        fence_id: u64,
        desc_index: u16,
        len: u32,
    ) -> bool {
        // In case the fence is signaled immediately after creation, don't add a return
        // FenceDescriptor.
        let mut fence_state = self.fence_state.lock().unwrap();
        if fence_id > *fence_state.completed_fences.get(&ring).unwrap_or(&0) {
            fence_state.descs.push(FenceDescriptor {
                ring,
                fence_id,
                desc_index,
                len,
            });

            false
        } else {
            true
        }
    }

    /// Creates a blob resource using rutabaga.
    pub fn resource_create_blob(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        vecs: Vec<(GuestAddress, usize)>,
        mem: &GuestMemoryMmap,
    ) -> VirtioGpuResult {
        let mut rutabaga_iovecs = None;

        if resource_create_blob.blob_flags & VIRTIO_GPU_BLOB_FLAG_CREATE_GUEST_HANDLE != 0 {
            panic!("GUEST_HANDLE unimplemented");
        } else if resource_create_blob.blob_mem != VIRTIO_GPU_BLOB_MEM_HOST3D {
            rutabaga_iovecs =
                Some(sglist_to_rutabaga_iovecs(&vecs[..], mem).map_err(|_| ErrUnspec)?);
        }

        self.rutabaga.resource_create_blob(
            ctx_id,
            resource_id,
            resource_create_blob,
            rutabaga_iovecs,
            None,
        )?;

        let resource = VirtioGpuResource::new(resource_id, 0, 0, resource_create_blob.size);

        // Rely on rutabaga to check for duplicate resource ids.
        self.resources.insert(resource_id, resource);
        Ok(self.result_from_query(resource_id))
    }

    /// Uses the hypervisor to map the rutabaga blob resource.
    ///
    /// When sandboxing is disabled, external_blob is unset and opaque fds are mapped by
    /// rutabaga as ExternalMapping.
    /// When sandboxing is enabled, external_blob is set and opaque fds must be mapped in the
    /// hypervisor process by Vulkano using metadata provided by Rutabaga::vulkan_info().
    /*
    pub fn resource_map_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
        offset: u64,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let map_info = self.rutabaga.map_info(resource_id).map_err(|_| ErrUnspec)?;

        if let Ok(export) = self.rutabaga.export_blob(resource_id) {
            if export.handle_type != RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD {
                let prot = match map_info & RUTABAGA_MAP_ACCESS_MASK {
                    RUTABAGA_MAP_ACCESS_READ => libc::PROT_READ,
                    RUTABAGA_MAP_ACCESS_WRITE => libc::PROT_WRITE,
                    RUTABAGA_MAP_ACCESS_RW => libc::PROT_READ | libc::PROT_WRITE,
                    _ => panic!("unexpected prot mode for mapping"),
                };

                if offset + resource.size > shm_region.size as u64 {
                    log::error!("mapping DOES NOT FIT");
                }
                let addr = shm_region.host_addr + offset;
                log::debug!(
                    "mapping: host_addr={:x}, addr={:x}, size={}",
                    shm_region.host_addr, addr, resource.size
                );
                let ret = unsafe {
                    libc::mmap(
                        addr as *mut libc::c_void,
                        resource.size as usize,
                        prot,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        export.os_handle.as_raw_fd(),
                        0 as libc::off_t,
                    )
                };
                if ret == libc::MAP_FAILED {
                    return Err(ErrUnspec);
                }
            } else {
                return Err(ErrUnspec);
            }
        } else {
            return Err(ErrUnspec);
        }

        resource.shmem_offset = Some(offset);
        // Access flags not a part of the virtio-gpu spec.
        Ok(OkMapInfo {
            map_info: map_info & RUTABAGA_MAP_CACHE_MASK,
        })
    }*/

    /// Uses the hypervisor to unmap the blob resource.
    /*
    pub fn resource_unmap_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let shmem_offset = resource.shmem_offset.ok_or(ErrUnspec)?;

        let addr = shm_region.host_addr + shmem_offset;

        let ret = unsafe {
            libc::mmap(
                addr as *mut libc::c_void,
                resource.size as usize,
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
                -1,
                0_i64,
            )
        };
        if ret == libc::MAP_FAILED {
            panic!("UNMAP failed");
        }

        resource.shmem_offset = None;

        Ok(OkNoData)
    }*/

    fn process_gpu_command(
        &mut self,
        mem: &GuestMemoryMmap,
        hdr: virtio_gpu_ctrl_hdr,
        cmd: GpuCommand,
        reader: &mut Reader,
    ) -> VirtioGpuResult {
        self.force_ctx_0();

        match cmd {
            GpuCommand::GetDisplayInfo(_) => {
                panic!("virtio_gpu: GpuCommand::GetDisplayInfo unimplemented");
            }
            GpuCommand::ResourceCreate2d(info) => {
                let resource_id = info.resource_id;

                let resource_create_3d = ResourceCreate3D {
                    target: RUTABAGA_PIPE_TEXTURE_2D,
                    format: info.format,
                    bind: RUTABAGA_PIPE_BIND_RENDER_TARGET,
                    width: info.width,
                    height: info.height,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: 0,
                };

                self.resource_create_3d(resource_id, resource_create_3d)
            }
            GpuCommand::ResourceUnref(info) => self.unref_resource(info.resource_id),
            GpuCommand::SetScanout(_info) => {
                panic!("virtio_gpu: GpuCommand::SetScanout unimplemented");
            }
            GpuCommand::ResourceFlush(info) => self.flush_resource(info.resource_id),
            GpuCommand::TransferToHost2d(info) => {
                let resource_id = info.resource_id;
                let transfer = Transfer3D::new_2d(info.r.x, info.r.y, info.r.width, info.r.height);
                self.transfer_write(0, resource_id, transfer)
            }
            GpuCommand::ResourceAttachBacking(info) => {
                let available_bytes = reader.available_bytes();
                if available_bytes != 0 {
                    let entry_count = info.nr_entries as usize;
                    let mut vecs = Vec::with_capacity(entry_count);
                    for _ in 0..entry_count {
                        match reader.read_obj::<virtio_gpu_mem_entry>() {
                            Ok(entry) => {
                                let addr = GuestAddress(entry.addr);
                                let len = entry.length as usize;
                                vecs.push((addr, len))
                            }
                            Err(_) => return Err(GpuResponse::ErrUnspec),
                        }
                    }
                    self.attach_backing(info.resource_id, mem, vecs)
                } else {
                    log::error!("missing data for command {:?}", cmd);
                    Err(GpuResponse::ErrUnspec)
                }
            }
            GpuCommand::ResourceDetachBacking(info) => self.detach_backing(info.resource_id),
            GpuCommand::UpdateCursor(_info) => {
                panic!("virtio_gpu: GpuCommand:UpdateCursor unimplemented");
            }
            GpuCommand::MoveCursor(_info) => {
                panic!("virtio_gpu: GpuCommand::MoveCursor unimplemented");
            }
            GpuCommand::ResourceAssignUuid(info) => {
                let resource_id = info.resource_id;
                self.resource_assign_uuid(resource_id)
            }
            GpuCommand::GetCapsetInfo(info) => self.get_capset_info(info.capset_index),
            GpuCommand::GetCapset(info) => self.get_capset(info.capset_id, info.capset_version),

            GpuCommand::CtxCreate(info) => {
                let context_name: Option<String> = String::from_utf8(info.debug_name.to_vec()).ok();
                self.create_context(hdr.ctx_id, info.context_init, context_name.as_deref())
            }
            GpuCommand::CtxDestroy(_info) => self.destroy_context(hdr.ctx_id),
            GpuCommand::CtxAttachResource(info) => {
                self.context_attach_resource(hdr.ctx_id, info.resource_id)
            }
            GpuCommand::CtxDetachResource(info) => {
                self.context_detach_resource(hdr.ctx_id, info.resource_id)
            }
            GpuCommand::ResourceCreate3d(info) => {
                let resource_id = info.resource_id;
                let resource_create_3d = ResourceCreate3D {
                    target: info.target,
                    format: info.format,
                    bind: info.bind,
                    width: info.width,
                    height: info.height,
                    depth: info.depth,
                    array_size: info.array_size,
                    last_level: info.last_level,
                    nr_samples: info.nr_samples,
                    flags: info.flags,
                };

                self.resource_create_3d(resource_id, resource_create_3d)
            }
            GpuCommand::TransferToHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                self.transfer_write(ctx_id, resource_id, transfer)
            }
            GpuCommand::TransferFromHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                self.transfer_read(ctx_id, resource_id, transfer, None)
            }
            GpuCommand::CmdSubmit3d(info) => {
                if reader.available_bytes() != 0 {
                    let num_in_fences = info.num_in_fences as usize;
                    let cmd_size = info.size as usize;
                    let mut cmd_buf = vec![0; cmd_size];
                    let mut fence_ids: Vec<u64> = Vec::with_capacity(num_in_fences);

                    for _ in 0..num_in_fences {
                        match reader.read_obj::<u64>() {
                            Ok(fence_id) => {
                                fence_ids.push(fence_id);
                            }
                            Err(_) => return Err(GpuResponse::ErrUnspec),
                        }
                    }

                    if reader.read_exact(&mut cmd_buf[..]).is_ok() {
                        self.submit_command(hdr.ctx_id, &mut cmd_buf[..], &fence_ids)
                    } else {
                        Err(GpuResponse::ErrInvalidParameter)
                    }
                } else {
                    // Silently accept empty command buffers to allow for
                    // benchmarking.
                    Ok(GpuResponse::OkNoData)
                }
            }
            GpuCommand::ResourceCreateBlob(info) => {
                let resource_id = info.resource_id;
                let ctx_id = hdr.ctx_id;

                let resource_create_blob = ResourceCreateBlob {
                    blob_mem: info.blob_mem,
                    blob_flags: info.blob_flags,
                    blob_id: info.blob_id,
                    size: info.size,
                };

                let entry_count = info.nr_entries;
                if reader.available_bytes() == 0 && entry_count > 0 {
                    return Err(GpuResponse::ErrUnspec);
                }

                let mut vecs = Vec::with_capacity(entry_count as usize);
                for _ in 0..entry_count {
                    match reader.read_obj::<virtio_gpu_mem_entry>() {
                        Ok(entry) => {
                            let addr = GuestAddress(entry.addr);
                            let len = entry.length as usize;
                            vecs.push((addr, len))
                        }
                        Err(_) => return Err(GpuResponse::ErrUnspec),
                    }
                }

                self.resource_create_blob(ctx_id, resource_id, resource_create_blob, vecs, mem)
            }
            GpuCommand::SetScanoutBlob(_info) => {
                panic!("virtio_gpu: GpuCommand::SetScanoutBlob unimplemented");
            }
            GpuCommand::ResourceMapBlob(info) => {
                /*
                let resource_id = info.resource_id;
                let offset = info.offset;
                self.resource_map_blob(resource_id, &self.shm_region, offset)
                */
                log::trace!("{cmd:?} not implemented!");
                Err(ErrUnspec)
            }
            GpuCommand::ResourceUnmapBlob(info) => {
                /*
                let resource_id = info.resource_id;
                self.resource_unmap_blob(resource_id, &self.shm_region)
                */
                log::trace!("{cmd:?} not implemented!");
                Err(ErrUnspec)
            }
        }
    }

    pub fn process_queue(&mut self, vring: &VringRwLock, mem: &GuestMemoryMmap) {
        let mut used_any = false;

        while let Some(head) = {
            let mut locked_vring = vring.get_mut();
            let queue = locked_vring.get_queue_mut();
            queue.pop_descriptor_chain(mem)
        } {
            let mut reader = Reader::new(&mem, head.clone())
                .map_err(GpuError::QueueReader)
                .unwrap();
            let mut writer = Writer::new(&mem, head.clone())
                .map_err(GpuError::QueueWriter)
                .unwrap();

            let mut resp = Err(GpuResponse::ErrUnspec);
            let mut gpu_cmd = None;
            let mut ctrl_hdr = None;
            let mut len = 0;

            match GpuCommand::decode(&mut reader) {
                Ok((hdr, cmd)) => {
                    resp = self.process_gpu_command(&mem, hdr, cmd, &mut reader);
                    ctrl_hdr = Some(hdr);
                    gpu_cmd = Some(cmd);
                }
                Err(e) => log::debug!("descriptor decode error: {:?}", e),
            }

            let mut gpu_response = match resp {
                Ok(gpu_response) => gpu_response,
                Err(gpu_response) => {
                    log::debug!("{:?} -> {:?}", gpu_cmd, gpu_response);
                    gpu_response
                }
            };

            let mut add_to_queue = true;

            if writer.available_bytes() != 0 {
                let mut fence_id = 0;
                let mut ctx_id = 0;
                let mut flags = 0;
                let mut ring_idx = 0;
                if let Some(_cmd) = gpu_cmd {
                    let ctrl_hdr = ctrl_hdr.unwrap();
                    if ctrl_hdr.flags & VIRTIO_GPU_FLAG_FENCE != 0 {
                        flags = ctrl_hdr.flags;
                        fence_id = ctrl_hdr.fence_id;
                        ctx_id = ctrl_hdr.ctx_id;
                        ring_idx = ctrl_hdr.ring_idx;

                        let fence = RutabagaFence {
                            flags,
                            fence_id,
                            ctx_id,
                            ring_idx,
                        };
                        gpu_response = match self.create_fence(fence) {
                            Ok(_) => gpu_response,
                            Err(fence_resp) => {
                                log::warn!("create_fence {} -> {:?}", fence_id, fence_resp);
                                fence_resp
                            }
                        };
                    }
                }

                // Prepare the response now, even if it is going to wait until
                // fence is complete.
                match gpu_response.encode(flags, fence_id, ctx_id, ring_idx, &mut writer) {
                    Ok(l) => len = l,
                    Err(e) => log::debug!("ctrl queue response encode error: {:?}", e),
                }

                if flags & VIRTIO_GPU_FLAG_FENCE != 0 {
                    let ring = match flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                        0 => VirtioGpuRing::Global,
                        _ => VirtioGpuRing::ContextSpecific { ctx_id, ring_idx },
                    };

                    add_to_queue = self.process_fence(ring, fence_id, head.head_index(), len);
                }
            }

            if add_to_queue {
                /*self.queue_ctl
                   .lock()
                   .unwrap()
                   .add_used(&mem, head.index, len);
                */
                used_any = true;
                vring.add_used(head.head_index(), len);
            }
        }

        log::debug!("gpu: process_queue exit");
        if used_any {
            if let Err(e) = vring.signal_used_queue() {
                log::error!("Failed to signal used queue!");
            }
        }
    }
}
