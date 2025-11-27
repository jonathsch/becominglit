

export default function Figure({
    image,
    caption,
}: {
    image: React.ReactNode;
    caption?: React.ReactNode;
}) {
    return (
        <figure className="w-full flex flex-col gap-2 items-center">
            <div className="*:max-h-[35rem] *:object-contain flex justify-center w-full">
                {image}
            </div>
            {caption && (
                <figcaption className="text-center text-zinc-700 dark:text-zinc-300">
                    {caption}
                </figcaption>
            )}
        </figure>
    );
}