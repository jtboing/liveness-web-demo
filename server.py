import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import time

import sys

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

from liveness import Liveness

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

class VideoAnalyzeTrack(MediaStreamTrack):
    """
    A video stream track that will apply the liveness analysis
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.label = "label_vat"

    async def recv(self):
        #TO-DO implement liveness analysis
        frame = await self.track.recv()
        image = frame.to_ndarray(format="bgr24")

        l = Liveness(image)
        new_image = l.start()

        self.label = l.label

        del l

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(new_image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connectionstate is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "false":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track video received")

        local_video = VideoAnalyzeTrack(track)
        pc.addTrack(local_video)

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                pass
                # if isinstance(message, str) and message.startswith("ping"):
                if isinstance(message, str):
                    channel.send("result: " + str(local_video.label))
        
        @track.on("ended")
        async def on_ended():
            log_info("Track video ended")

    
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connection
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
#async def factory():
    parser = argparse.ArgumentParser(
        description="WebRTC video / data-channels and liveness detection demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    #return app
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
