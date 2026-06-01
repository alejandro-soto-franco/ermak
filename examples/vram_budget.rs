//! Print the device-memory budget derived from free VRAM (feature `gpu`).
fn main() {
    #[cfg(feature = "gpu")]
    {
        match ermak::gpu::device_budget(0.5) {
            Ok(b) => println!(
                "device budget (50% of free VRAM): {} MiB",
                b.cap_bytes() / 1024 / 1024
            ),
            Err(e) => println!("no device budget: {e}"),
        }
    }
    #[cfg(not(feature = "gpu"))]
    println!("build with --features gpu");
}
