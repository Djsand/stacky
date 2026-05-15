from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .protocol import BodyCommand, BodyEvent


@dataclass(eq=False)
class StackChanConnection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    async def send(self, command: BodyCommand) -> None:
        self.writer.write((command.to_json() + "\n").encode("utf-8"))
        await asyncio.wait_for(self.writer.drain(), timeout=1.0)

    def close(self) -> None:
        self.writer.close()


class StackChanBodyHub:
    """Line-delimited JSON body hub for StackChan custom firmware.

    Audio frames are supported as base64 payloads in v1. This keeps the first
    firmware path debuggable; a binary side-channel can be added once the body
    is stable.
    """

    def __init__(self) -> None:
        self.connections: set[StackChanConnection] = set()
        self.events: asyncio.Queue[BodyEvent] = asyncio.Queue()
        self._server: asyncio.AbstractServer | None = None

    async def start(self, host: str, port: int) -> None:
        self._server = await asyncio.start_server(self._handle_client, host, port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for connection in list(self.connections):
            connection.close()
        self.connections.clear()

    async def broadcast(self, command: BodyCommand) -> None:
        stale: list[StackChanConnection] = []
        for connection in self.connections:
            try:
                await connection.send(command)
            except OSError:
                stale.append(connection)
        for connection in stale:
            self.connections.discard(connection)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection = StackChanConnection(reader, writer)
        self.connections.add(connection)
        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw:
                    break
                event = BodyEvent.from_json(raw.decode("utf-8"))
                await self.events.put(event)
        finally:
            self.connections.discard(connection)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
