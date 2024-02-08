use std::sync::{Arc, Mutex};
use std::sync::mpsc::Sender;
use crossbeam_channel::unbounded;
use log::warn;
use vhost::vhost_user::VhostUserProtocolFeatures;
use vhost::vhost_user::VhostUserVirtioFeatures;
use vhost_user_backend::{VhostUserBackendMut, VringRwLock, VringT};
use virtio_bindings::virtio_config::VIRTIO_F_VERSION_1;
use virtio_bindings::virtio_gpu::{
    VIRTIO_GPU_F_CONTEXT_INIT, VIRTIO_GPU_F_RESOURCE_BLOB, VIRTIO_GPU_F_VIRGL,
};
use vm_memory::{Bytes, GuestAddress, GuestAddressSpace, GuestMemory, GuestMemoryAtomic, GuestMemoryMmap};
use vmm_sys_util::epoll::EventSet;
use crate::virtio_gpu::VirtioGpu;

const CONTROLQ_ID: u16 = 0;
const EVENT_ID_CURSORQ: u16 = 1;

#[derive(Clone)]
pub struct VirtioShmRegion {
    pub host_addr: u64,
    pub guest_addr: u64,
    pub size: usize,
}

pub struct GpuBackend {
    event_idx: bool,
    mem: Option<GuestMemoryAtomic<GuestMemoryMmap>>,
    // TODO: we cannot have this here because VirtioGpu is not Send
    //gpu: Arc<Mutex<Option<VirtioGpu>>>,
}

impl GpuBackend {
    pub fn new() -> Self {
        GpuBackend {
            event_idx: false,
            mem: None,
           // gpu: Arc::new(Mutex::new(None)),
        }
    }
}

impl VhostUserBackendMut for GpuBackend {
    type Bitmap = ();
    type Vring = VringRwLock;

    fn num_queues(&self) -> usize {
        2
    }

    fn max_queue_size(&self) -> usize {
        256
    }

    fn features(&self) -> u64 {
        1 << VIRTIO_F_VERSION_1
            | 1 << VIRTIO_GPU_F_VIRGL
            | 1 << VIRTIO_GPU_F_RESOURCE_BLOB
            | 1 << VIRTIO_GPU_F_CONTEXT_INIT
            | VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits()
    }

    fn protocol_features(&self) -> VhostUserProtocolFeatures {
        VhostUserProtocolFeatures::MQ | VhostUserProtocolFeatures::CONFIG
    }

    fn set_event_idx(&mut self, enabled: bool) {
        self.event_idx = enabled;
    }

    fn update_memory(
        &mut self,
        mem: GuestMemoryAtomic<GuestMemoryMmap<Self::Bitmap>>,
    ) -> std::io::Result<()> {
        if self.mem.is_some() {
            panic!("Changed memory after starting not supported!");
        }

        self.mem = Some(mem);

        Ok(())
    }

    fn handle_event(
        &mut self,
        device_event: u16,
        evset: EventSet,
        vrings: &[Self::Vring],
        thread_id: usize,
    ) -> std::io::Result<()> {
        assert_eq!(thread_id, 0);
        let Some(ref mem) = self.mem else {
            log::error!("Cannot handle_event: we don't have shared memory!");
            return Ok(())
        };

        /*
        let Some(ref gpu) = self.gpu.lock() else {
            //FIXME: what is the shm_region
            let gpu = VirtioGpu::new(vrings[CONTROLQ_ID].clone());
            self.gpu = Arc::new(Mutex::new(Some(gpu)));

            self.gpu.lock().as_mut().unwrap()
        };*/

        match device_event {
            CONTROLQ_ID => {

            },
            CURSORQ_ID => {
                log::trace!("Ignoring CURSORQ: not implemented");
            }
            device_event => {
                log::warn!("unhandled device_event: {}", device_event);
                panic!("TODO error");
                //return Err(Error::HandleEventUnknown.into());
            }
        }

        Ok(())
    }
}
