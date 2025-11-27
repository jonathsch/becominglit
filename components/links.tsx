import type { LinkEntry } from '@/types/types';
import Link from 'next/link';
import { DynamicIcon } from 'lucide-react/dynamic';

interface Props {
    links: LinkEntry[];
}

export default function Links({ links }: Props) {
    return (
        < div className="flex flex-row flex-wrap justify-center gap-2" >
            {
                links.map((link) => (
                    <Link
                        key={link.name}
                        href={link.url}
                        className="flex flex-row bg-zinc-800 dark:bg-zinc-200 text-white dark:text-zinc-900 rounded-full gap-2 items-center text-lg px-5 py-2 hover:bg-zinc-950 dark:hover:bg-zinc-50 hover:no-underline"
                    >
                        {link.icon}
                        <span>{link.name}</span>
                    </Link>
                ))
            }
        </div >
    );
}