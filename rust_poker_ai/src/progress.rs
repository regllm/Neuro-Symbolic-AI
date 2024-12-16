use indicatif::{ProgressStyle};


pub fn create_progress_style() -> ProgressStyle {
    ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap()
}
