use indicatif::{ProgressBar, ProgressStyle};


pub fn create_progress_style() -> ProgressStyle {
    ProgressStyle::default_bar().template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta} left)").unwrap()
}

pub fn new(total: u64) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(create_progress_style());
    bar
}
