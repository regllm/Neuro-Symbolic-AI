use clap::Parser;


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Run experiments
    #[arg(long, default_value_t = false)]
    pub exp: bool,

    /// Use short deck for quick tests
    #[arg(long, default_value_t = false)]
    pub short: bool,

    /// Run in a table-server mode
    #[arg(long, default_value_t = false)]
    pub table: bool,

    /// Run in a table-server mode
    #[arg(long, default_value_t = String::from("./output"))]
    pub table_path: String,

    /// Run in a table-server mode
    #[arg(long, default_value_t = String::from("127.0.0.1"))]
    pub host: String,

    /// Run in a table-server mode
    #[arg(long, default_value_t = String::from("8989"))]
    pub port: String,
}


pub fn get_args() -> Args {
    Args::parse()
}
