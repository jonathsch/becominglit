
interface YouTubeVideoProps {
    videoId: string;
    aspectRatio: any; // Default aspect ratio
}

export default function YouTubeVideo({ videoId, aspectRatio = 16 / 9 }: YouTubeVideoProps) {
    return (
        <div className="rounded-lg overflow-hidden w-full" style={{ aspectRatio: aspectRatio }}>
            <iframe
                className="w-full h-full"
                loading="lazy"
                src={"https://www.youtube.com/embed/" + videoId}
                title="YouTube video player"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen>
            </iframe>
        </div>
    )
}