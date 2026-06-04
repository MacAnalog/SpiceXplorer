# Vendored IHP `ihp-sg13g2` PDK — ngspice subset

This directory ships the **minimal subset** of the IHP Open PDK that SpiceXplorer's
ngspice flow needs, so the Docker image is self-contained (no host PDK, no 5 GB
EDA base image). It is **~2.4 MB** vs. the ~227 MB full PDK — we keep only
`libs.tech/ngspice/`:

```
ihp-sg13g2/libs.tech/
  ngspice/
    .spiceinit        # sets sourcepath + loads the OSDI compact models
    models/*.lib      # device model libraries (cornerMOSlv.lib, cornerRES.lib, …)
    osdi/*.osdi       # prebuilt OSDI compact models — x86-64 (used by OSDI_MODE=vendor)
  verilog-a/          # OSDI source (psp103, r3_cmc, mosvar) — compiled per-arch
    {psp103,r3_cmc,mosvar}/*.va    #   by openvaf when OSDI_MODE=compile (default)
```

The image's `docker/Dockerfile.backend` copies this tree to `/opt/pdk/` and points
`PDK_ROOT=/opt/pdk`, `PDK=ihp-sg13g2` at it.

## License

The IHP Open PDK is licensed under **Apache License 2.0** and is freely
redistributable. Every `*.lib` carries its per-file copyright header; the upstream
project is <https://github.com/IHP-GmbH/IHP-Open-PDK>. Do not strip the headers.

## Architecture note (the `*.osdi` files)

OSDI compact models are architecture-specific compiled binaries. The backend
image handles this with the `OSDI_MODE` build arg:

- **`compile` (default):** `docker/Dockerfile.backend` builds openvaf and compiles
  the `verilog-a/` sources **for the build's own architecture** — so the image is
  native on x86-64 **and** arm64 (incl. Apple silicon), no emulation. This is the
  recipe below, run automatically at build time.
- **`vendor`:** reuse the committed `osdi/*.osdi`, which are **x86-64 ELF**
  (ABI-matched to ngspice 45). Faster (no openvaf toolchain) but x86-64 only.

To regenerate the committed prebuilt `osdi/*.osdi` (e.g. to refresh `vendor` mode
or track a newer PDK), use the same recipe iic-osic-tools uses:

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

Then replace the `osdi/*.osdi` here. (The default `compile` mode runs exactly this
in a throwaway build stage; the committed `osdi/*.osdi` exist only for the optional
`vendor` fast-path, since the openvaf build needs a ~1 GB LLVM-18 + Rust toolchain.)
