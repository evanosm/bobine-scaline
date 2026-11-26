//! Programme natif de Scaline. Bobine le lance avec `--bobine-worker 1`, dans son dossier de
//! travail (`BOBINE_WORK_DIR`), lui écrit une requête JSON par ligne sur stdin et lit une ligne
//! JSON par réponse sur stdout (voir `protocol.rs`). Il lit son dossier de travail et les
//! modèles de son paquet, écrit seulement dans son dossier de travail, et s'arrête quand stdin
//! se ferme. stderr va au journal de Bobine.

use std::io::{BufRead, Write};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) != Some("--bobine-worker") {
        eprintln!("scaline-worker : à lancer par Bobine (--bobine-worker 1)");
        std::process::exit(2);
    }
    if args.get(2).map(String::as_str) != Some("1") {
        eprintln!("scaline-worker : seul le protocole 1 est connu");
        std::process::exit(2);
    }
    let Some(work_dir) = std::env::var_os("BOBINE_WORK_DIR").map(PathBuf::from) else {
        eprintln!("scaline-worker : BOBINE_WORK_DIR manquant");
        std::process::exit(2);
    };
    let Some(models) = scaline_worker::protocol::models_dir() else {
        eprintln!("scaline-worker : dossier des modèles introuvable");
        std::process::exit(2);
    };
    eprintln!("scaline-worker {} prêt", env!("CARGO_PKG_VERSION"));

    let mut state = scaline_worker::protocol::State::new(work_dir, models);
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let mut broken = false;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        scaline_worker::protocol::handle_line(&mut state, &line, &mut |value| {
            // Une ligne par message, vidée tout de suite : Bobine lit au fil de l'eau.
            if writeln!(out, "{value}").and_then(|()| out.flush()).is_err() {
                broken = true;
            }
        });
        if broken {
            break;
        }
    }
}
