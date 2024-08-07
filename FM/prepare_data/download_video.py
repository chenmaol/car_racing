# download video from youtube

from youtubesearchpython import VideosSearch
import yt_dlp
import json


videosSearch = VideosSearch('WRC Generation Gameplay', limit = 100)


def format_selector(ctx):
    """ Select the best video and the best audio that won't result in an mkv.
    NOTE: This is just an example and does not handle all cases """

    # formats are already sorted worst to best
    formats = ctx.get('formats')

    # filter_formats = []
    # for format in formats:
    #     if 'format_note' in format:
    #         filter_formats.append(format)
    # formats = filter_formats
    # acodec='none' means there is no audio
    best_video = next(f for f in formats
                      if f['vcodec'] != 'none' and f['acodec'] == 'none' and f['ext'] == 'mp4')

    # find compatible audio extension
    audio_ext = {'mp4': 'm4a'}[best_video['ext']]
    # vcodec='none' means there is no video
    best_audio = next(f for f in formats if (
        f['acodec'] != 'none' and f['vcodec'] == 'none' and f['ext'] == audio_ext))

    # These are the minimum required fields for a merged format
    yield {
        'format_id': f'{best_video["format_id"]}+{best_audio["format_id"]}',
        'ext': best_video['ext'],
        'requested_formats': [best_video, best_audio],
        # Must be + separated list of protocols
        'protocol': f'{best_video["protocol"]}+{best_audio["protocol"]}'
    }


ydl_opts = {
    'format': 'mp4',
}
videosSearch.next()

urls = [
'https://www.youtube.com/watch?v=dEKHItSPGEo&t=172s',
'https://www.youtube.com/watch?v=lSAduN50LcE&t=659s',
'https://www.youtube.com/playlist?list=PL6Z5cLfwdiFRkD7Y-rAkSbEOHGp_-O2oZ',
'https://www.youtube.com/watch?v=FiWEeMHGzaU&t=1s',
'https://www.youtube.com/@ghostxinsanity8787/search?query=WRC%20Generations',
]
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    # while True:
    #     results = videosSearch.result()['result']
    #     urls = [result['link'] for result in results]
    #     videosSearch.next()
    #
    #     # info = ydl.extract_info(urls[0], download=False)
    #     #
    #     # # ℹ️ ydl.sanitize_info makes the info json-serializable
    #     # print(json.dumps(ydl.sanitize_info(info)))

        ydl.download(urls)
