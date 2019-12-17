fn main() {
    if cfg!(not(target_os = "windows")) && cfg!(feature = "llvm") {
        println!("cargo:rustc-link-lib=dylib=ffi");
    }
}
