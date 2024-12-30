use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[clap(version, about, long_about = None)]
pub struct App {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Cluster(ClusterArgs),
    Table(TableArgs),
    Exp,
}

#[derive(Debug, Args)]
pub struct ClusterArgs {
    /// Whether to cluster hands out of the short deck with only 10, J, Q, K, A
    #[arg(long, default_value_t = false)]
    pub short: bool,
}

#[derive(Debug, Args)]
pub struct TableArgs {
    /// A path to load lookup table text files from.
    #[arg(long, default_value_t = String::from("./output"))]
    pub input: String,

    /// A hostname to serve the lookup table server at.
    #[arg(long, default_value_t = String::from("127.0.0.1"))]
    pub host: String,

    /// A port number to serve the lookup table server at.
    #[arg(long, default_value_t = String::from("8989"))]
    pub port: String,
}

pub fn get_app() -> App {
    App::parse()
}

pub fn get_command() -> Command {
    get_app().command
}

pub fn get_cluster_args() -> Option<ClusterArgs> {
    if let Command::Cluster(sub_args) = get_command() {
        Some(sub_args)
    } else {
        None
    }
}

pub fn get_table_args() -> Option<TableArgs> {
    if let Command::Table(sub_args) = get_command() {
        Some(sub_args)
    } else {
        None
    }
}
