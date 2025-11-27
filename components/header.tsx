import type { Author, Note, LinkEntry } from '../types/types';
import Authors from './authors';
import Links from './links';

interface Props {
    title: string;
    authors: Author[];
    conference?: string;
    notes?: Note[];
    links: LinkEntry[];
}

export default function Header({ title, authors, conference, notes, links }: Props) {
    return (
        <header className="flex flex-col gap-10 items-center mb-6">
            <h1>{title}</h1>
            <div className="flex flex-col gap-6 items-center">
                <Authors authors={authors} />
                {conference && <p className="text-center font-bold">{conference}</p>}
                {/* {notes && <Notes notes={notes} />} */}
                {links && <Links links={links} />}
            </div>
        </header>
    )
}