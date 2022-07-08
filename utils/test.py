import asyncio
from binance import Client, AsyncClient, BinanceSocketManager


async def main():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    # start any sockets here, i.e a trade socket
    ks = bm.kline_socket('BNBBTC', interval=Client.KLINE_INTERVAL_15MINUTE)
    # then start receiving messages
    async with ks as tscm:
        while True:
            res = await tscm.recv()
            print(res)


    await client.close_connection()

if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())