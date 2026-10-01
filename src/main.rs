use std::process::ExitCode;

use axum::serve as axum_serve;
use serde_json::to_string_pretty;
use tokio::net::TcpListener;
use uuid::Uuid;

use dag_path_cover::{
    logging::init, model::GraphInput, query::router, solve_dag_path_cover, Config,
};

fn usage() -> &'static str {
    "usage:\n  dag-path-cover serve [config.toml]\n  dag-path-cover solve <graph.json> [max-vertices]"
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("serve") => serve(args.get(2).map(String::as_str)).await,
        Some("solve") => solve(args.get(2), args.get(3)),
        Some("--help") | Some("-h") | Some("help") | None => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("unknown command `{command}`\n{}", usage());
            ExitCode::from(2)
        }
    }
}

async fn serve(config_path: Option<&str>) -> ExitCode {
    let config = match Config::load(config_path) {
        Ok(config) => config,
        Err(error) => {
            eprintln!(
                "configuration error [{}]: {}",
                error.code(),
                error.message()
            );
            return ExitCode::FAILURE;
        }
    };

    init(&config.log_level);

    let address = config.socket_addr();
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        %address,
        max_vertices = config.max_vertices,
        "starting DAG path cover HTTP service"
    );

    let listener = match TcpListener::bind(address).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, %address, "failed to bind listener");
            return ExitCode::FAILURE;
        }
    };

    if let Err(error) = axum_serve(listener, router(config)).await {
        tracing::error!(%error, "HTTP server failed");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn solve(path: Option<&String>, max_vertices_arg: Option<&String>) -> ExitCode {
    let Some(path) = path else {
        eprintln!("missing graph file\n{}", usage());
        return ExitCode::from(2);
    };

    let config = match Config::load(None) {
        Ok(config) => config,
        Err(error) => {
            eprintln!(
                "configuration error [{}]: {}",
                error.code(),
                error.message()
            );
            return ExitCode::FAILURE;
        }
    };
    init(&config.log_level);

    let max_vertices = match max_vertices_arg {
        Some(value) => match value.parse::<usize>() {
            Ok(value) if value > 0 && value <= 2000 => value,
            _ => {
                eprintln!("max-vertices must be a number from 1 through 2000");
                return ExitCode::from(2);
            }
        },
        None => config.max_vertices,
    };

    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("failed to read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let input: GraphInput = match serde_json::from_str(&contents) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("invalid JSON graph: {error}");
            return ExitCode::FAILURE;
        }
    };

    let run_id = format!("cli-{}", Uuid::new_v4());
    match solve_dag_path_cover(&run_id, &input, max_vertices) {
        Ok(response) => {
            match to_string_pretty(&response) {
                Ok(json) => println!("{json}"),
                Err(error) => {
                    eprintln!("failed to serialize response: {error}");
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("solve failed [{}]: {}", error.code(), error.message());
            match to_string_pretty(&error) {
                Ok(json) => eprintln!("{json}"),
                Err(serialize_error) => {
                    eprintln!("failed to serialize error details: {serialize_error}")
                }
            }
            ExitCode::FAILURE
        }
    }
}
