use std::mem::MaybeUninit;
use std::os::unix::net::UnixStream;
use vhost::vhost_user::connection::Endpoint;
use vhost::vhost_user::message::{BackendReq, Req, VhostUserMsgHeader};
use vm_memory::ByteValued;
use crate::protocol::{GpuResponse, virtio_gpu_resp_display_info};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VhostGpuReq {
    _FIRST = 0,
    /// Get the supported protocol features bitmask.
    GET_PROTOCOL_FEATURES = 1,
    /// Enable protocol features using a bitmask.
    SET_PROTOCOL_FEATURES,
    /// Get the preferred display configuration.
    GET_DISPLAY_INFO,
    /// Set/show the cursor position.
    CURSOR_POS,
    /// Set/hide the cursor.
    CURSOR_POS_HIDE,
    /// Set the scanout resolution.
    /// To disable a scanout, the dimensions width/height are set to 0.
    SCANOUT,
    /// Update the scanout content. The data payload contains the graphical bits.
    /// The display should be flushed and presented.
    UPDATE,
    /// Set the scanout resolution/configuration, and share a DMABUF file descriptor for the scanout content,
    /// which is passed as ancillary data.
    /// To disable a scanout, the dimensions width/height are set to 0, there is no file descriptor passed.
    DMABUF_SCANOUT,
    /// The display should be flushed and presented according to updated region from VhostUserGpuUpdate.
    // Note: there is no data payload, since the scanout is shared thanks to DMABUF,
    // that must have been set previously with VHOST_USER_GPU_DMABUF_SCANOUT.
    DMABUF_UPDATE,
    /// Retrieve the EDID data for a given scanout.
    /// This message requires the VHOST_USER_GPU_PROTOCOL_F_EDID protocol feature to be supported.
    GET_EDID,
    /// Same as VHOST_USER_GPU_DMABUF_SCANOUT, but also sends the dmabuf modifiers appended to the message,
    /// which were not provided in the other message.
    /// This message requires the VHOST_USER_GPU_PROTOCOL_F_DMABUF2 protocol feature to be supported.
    VHOST_USER_GPU_DMABUF_SCANOUT2,
    _LAST,
}

pub struct GpuFrontendConnection {
    endpoint: Endpoint<VhostGpuReq>,
}

impl From<VhostGpuReq> for u32 {
    fn from(req: VhostGpuReq) -> u32 {
        req as u32
    }
}

impl Req for VhostGpuReq {
    fn is_valid(value: u32) -> bool {
        (value > VhostGpuReq::_FIRST as u32) && (value < VhostGpuReq::_LAST as u32)
    }
}

const MSG_FLAGS: u32 = 0;

impl GpuFrontendConnection {
    pub fn from_stream(stream: UnixStream) -> Self {
        Self {
            endpoint: Endpoint::from_stream(stream)
        }
    }

    pub fn get_display_info(&mut self) -> Vec<(u32, u32, bool)> {
        self.endpoint.send_header(&VhostUserMsgHeader::new(
            VhostGpuReq::GET_DISPLAY_INFO,
            MSG_FLAGS,
            0,
        ), None).expect("TODO failed send_header");

        let (hdr, response, files) =
            self.endpoint.recv_body().expect("failed to recv");
        let response: virtio_gpu_resp_display_info = response;
        assert_eq!(hdr.get_code().expect("TODO get code failed"), VhostGpuReq::GET_DISPLAY_INFO);

        let displays = response.pmodes.iter().map(|display|
            (display.r.width, display.r.height, display.enabled == 1)
        ).collect();
        dbg!(displays)
    }
}