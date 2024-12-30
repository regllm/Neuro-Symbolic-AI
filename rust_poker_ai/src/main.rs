mod cluster;
mod poker;
mod args;
mod cluster_main;
mod exp;
mod progress;
mod table_main;


fn main() {
    match args::get_command() {
        args::Command::Cluster(_) => {
            cluster_main::build_lut();
        }
        args::Command::Table(_) => {
            table_main::run_server();
        }
        args::Command::Exp => {
            exp::exp();
        }
   }

}
