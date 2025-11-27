import Image from 'next/image';

export default function CustomImage({
    src,
    alt,
    width,
    height,
    className = '',
}: {
    src: string;
    alt?: string;
    width?: number;
    height?: number;
    className?: string;
}) {
    return (
        <Image
            src={src}
            alt={alt || ''}
            width={width || 800}
            height={height || 600}
            className="rounded-lg max-h-[35rem] w-max object-contain !px-0"
        />
    );
}