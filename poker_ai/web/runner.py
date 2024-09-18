import os

import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from poker_ai.utils.demo import PokerDemo, load_lut, load_strategy


gid_cursor = 0
games = {}
poker_data = {}


def create_app():
    app = FastAPI()

    app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


app = create_app()


@app.get("/", response_class=HTMLResponse)
async def root():
    file_path = os.path.normpath(
        os.path.join(
            os.path.abspath(__file__), "../../../assets/demo.html"
        ),
    )

    with open(file_path) as f:
        content = f.read()
    return content


@app.post("/games")
def create_game(request: Request):
    global gid_cursor

    gid = gid_cursor
    gid_cursor += 1

    games[gid] = PokerDemo(**poker_data)
    events = games[gid].read_events()

    return {"gid": gid, "events": events}


@app.get("/games/{gid}")
def get_game(request: Request, gid: int):
    return {"state":games[gid].to_dict()}


class CreateActionRequest(BaseModel):
    action: str


@app.post("/games/{gid}")
def create_game_action(request: Request, gid: int, body: CreateActionRequest):
    action = body.action

    games[gid].play(action)
    events = games[gid].read_events()

    return {"events": events}


@click.command()
@click.option(
    "--host",
    default="localhost",
    type=str,
    help=(
        "The host to serve a web-based demo"
    ),
)
@click.option(
    "--port",
    default=3000,
    type=int,
    help=(
        "The port number to serve a web-based demo"
    )
)
@click.option(
    "--include_dumb_players/--no_dumb_players",
    default=True,
    help=(
        "Play with random agents."
    )
)
@click.option(
    "--n_players",
    default=6,
    help=(
        "The number of players."
    )
)
@click.option(
    "--low_card_rank",
    default=2,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--high_card_rank",
    default=14,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option('--lut_path', required=False, default=".", type=str)
@click.option('--strategy_path', required=False, default="./agent.joblib", type=str)
@click.option('--debug_quick_start/--no_debug_quick_start', default=False)
def run_web_app(
    host: str,
    port: int,
    include_dumb_players: bool,
    n_players: int,
    low_card_rank: int,
    high_card_rank: int,
    lut_path: str,
    strategy_path: str,
    debug_quick_start: bool = False
):
    """Start up web-based demo app to play against your poker AI.

    Example
    -------

    Usually you would call this from the `poker_ai` CLI. Alternatively you can
    call this method from this module directly from python.

    ```bash
    python -m poker_ai.web.runner                                       \
        --lut_path ./research/blueprint_algo                               \
        --strategy_path ./agent.joblib                                       \
        --no_debug_quick_start
    ```
    """

    if not debug_quick_start:
        strategy = load_strategy(strategy_path)
        lut = load_lut(lut_path, low_card_rank, high_card_rank)
    else:
        strategy = None
        lut = None

    global poker_data
    poker_data = {
        "n_players": n_players,
        "low_card_rank": low_card_rank,
        "high_card_rank": high_card_rank,
        "strategy": strategy,
        "lut": lut,
        "include_dumb_players": include_dumb_players,
    }

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    run_web_app()
