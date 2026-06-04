# Vendored IHP `ihp-sg13g2` PDK — ngspice subset

This directory ships the **minimal subset** of the IHP Open PDK that SpiceXplorer's
ngspice flow needs, so the Docker image is self-contained (no host PDK, no 5 GB
EDA base image). It is **~2.4 MB** vs. the ~227 MB full PDK — we keep only
`libs.tech/ngspice/`:

```
ihp-sg13g2/libs.tech/ngspice/
  .spiceinit          # sets sourcepath + loads the OSDI compact models
  models/*.lib        # device model libraries (cornerMOSlv.lib, cornerRES.lib, …)
  osdi/*.osdi         # prebuilt OSDI compact models (PSP103, r3_cmc, mosvar)
```

The image's `docker/Dockerfile.backend` copies this tree to `/opt/pdk/` and points
`PDK_ROOT=/opt/pdk`, `PDK=ihp-sg13g2` at it.

## License

The IHP Open PDK is licensed under **Apache License 2.0** and is freely
redistributable. Every `*.lib` carries its per-file copyright header; the upstream
project is <https://github.com/IHP-GmbH/IHP-Open-PDK>. Do not strip the headers.

## Architecture note (the `*.osdi` files)

The OSDI files are **x86-64 ELF** shared objects, ABI-matched to ngspice ≥45.
The image therefore targets **x86-64**. To rebuild them (e.g. for arm64, or to
track a newer PDK), compile from the PDK's Verilog-A source with **openvaf**
(the same recipe iic-osic-tools uses):

```bash
# 1. Build openvaf-reloaded (needs Rust + LLVM-18):
git clone --filter=blob:none https://github.com/arpadbuermen/OpenVAF.git
cd OpenVAF && git checkout 2e066436d985b05cf8e6563e936daf9ab875775a
cargo build --release --features llvm18 --bin openvaf-r   # -> target/release/openvaf-r

# 2. Compile the models (from <PDK>/libs.tech/verilog-a/, per openvaf-compile-va.sh):
openvaf --target_cpu generic -D__NGSPICE__ -o ../ngspice/osdi/psp103.osdi     psp103/psp103.va
openvaf --target_cpu generic -D__NGSPICE__ -o ../ngspice/osdi/psp103_nqs.osdi psp103/psp103_nqs.va
openvaf --target_cpu generic -D__NGSPICE__ -o ../ngspice/osdi/r3_cmc.osdi     r3_cmc/r3_cmc.va
openvaf --target_cpu generic -D__NGSPICE__ -o ../ngspice/osdi/mosvar.osdi     mosvar/mosvar.va
```

Then replace the `osdi/*.osdi` here. (We vendor the prebuilt binaries rather than
compiling them in the Dockerfile because openvaf needs a ~1 GB LLVM-18 + Rust
toolchain — a heavy build stage for a 2 MB output.)
