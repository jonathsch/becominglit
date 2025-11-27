
interface VideoProps {
    source: string;
    controls?: boolean;
}

export default function Video({ source, controls }: VideoProps) {
    return (
        <div className="w-full flex justify-center">
            <video
                className="w-full h-auto aspect-video rounded-lg"
                autoPlay
                controls={controls ?? false}
                muted
                loop
                playsInline
            >
                <source src={source} type="video/mp4" />
            </video>
        </div>
    )
}