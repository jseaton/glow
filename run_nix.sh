#!/bin/sh

RUST_BACKTRACE=1 LD_LIBRARY_PATH=/nix/store/3zq4pspk1imq2ai8z40q1cv8hxcfp4d8-vulkan-loader-1.3.211.0/lib:/nix/store/2xrgg25np25icgf69in2d2fln90ghqn0-libxkbcommon-1.4.0/lib:$LD_LIBRARY_PATH  nix-shell -p vulkan-loader libxkbcommon pkg-config cmake python3 jack2 alsa-lib glslang --command "./notify.sh $@"
