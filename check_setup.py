
import asyncio
from rooms.room import Room
from agents.random_agent import RandomAgent


async def main():
    room = Room(
        run_remote_room=False,
        room_name="first_local_game",
        max_matches=1
    )

    players = [
        RandomAgent(name=f"P{i}", log_directory=room.room_dir)
        for i in range(4)
    ]

    for player in players:
        room.connect_player(player)

    await room.run()
    print("Final scores:", room.final_scores)


if __name__ == "__main__":
    asyncio.run(main())
