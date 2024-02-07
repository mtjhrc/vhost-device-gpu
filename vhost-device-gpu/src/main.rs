//
// Copyright 2023 Linaro Ltd. All Rights Reserved.
// Leo Yan <leo.yan@linaro.org>
//
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause

mod backend;
mod worker;
mod descriptor_utils;
mod file_traits;
mod virtio_gpu;
mod protocol;
//mod virtio_gpu;

use clap::Parser;
use std::{
    any::Any,
    path::PathBuf,
    process::exit,
    sync::{Arc, RwLock},
};

use crate::backend::GpuBackend;
use thiserror::Error as ThisError;
use vhost_user_backend::VhostUserDaemon;
use vm_memory::{GuestMemoryAtomic, GuestMemoryMmap};

#[derive(Debug, ThisError)]
#[allow(unused)]
/// Errors related to vhost-device-input daemon.
pub(crate) enum Error {
    #[error("Event device file doesn't exists or can't be accessed")]
    AccessEventDeviceFile,
    #[error("Could not create backend: {0}")]
    CouldNotCreateBackend(std::io::Error),
    #[error("Could not create daemon: {0}")]
    CouldNotCreateDaemon(vhost_user_backend::Error),
    #[error("Could not register input event into vring epoll")]
    CouldNotRegisterInputEvent,
    #[error("Fatal error: {0}")]
    ServeFailed(vhost_user_backend::Error),
    #[error("Thread `{0}` panicked")]
    ThreadPanic(String, Box<dyn Any + Send>),
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Parser, Debug, PartialEq)]
#[clap(author, version, about, long_about = None)]
struct Args {
    // Location of vhost-user Unix domain socket.
    #[clap(short, long, value_name = "SOCKET")]
    socket_path: PathBuf,
}

// This is the public API through which an external program starts the
/// vhost-device-input backend server.
pub(crate) fn start_backend_server(socket: PathBuf) -> Result<()> {
    loop {
        let backend = Arc::new(RwLock::new(GpuBackend::new()));

        let mut daemon = VhostUserDaemon::new(
            String::from("vhost-device-gpu-backend"),
            Arc::clone(&backend),
            GuestMemoryAtomic::new(GuestMemoryMmap::new()),
        )
        .map_err(Error::CouldNotCreateDaemon)?;

        let handlers = daemon.get_epoll_handlers();
        daemon.serve(&socket).map_err(Error::ServeFailed)?;
    }
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    if let Err(e) = start_backend_server(args.socket_path) {
        log::error!("{e}");
        exit(1);
    }
}
