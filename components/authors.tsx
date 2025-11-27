import Link from "next/link"
import type { Author } from "../types/types";

interface Props {
    authors: Author[];
}

export default function Authors({ authors }: Props) {
    return (
        <div className="flex flex-row gap-x-8 gap-y-4 flex-wrap justify-center">
            {
                authors.map((author) => (
                    <div className="flex flex-col items-center text-center" key={author.name}>
                        <span className="text-xl flex flex-row">
                            {author.url ? (
                                <Link href={author.url} className="">
                                    {author.name}
                                </Link>
                            ) : (
                                author.name
                            )}
                            {author.notes && (
                                <sup className="text-xl">
                                    {author.notes.map(
                                        (note, index, array) =>
                                            note + (index < array.length - 1 ? "," : "")
                                    )}
                                </sup>
                            )}
                        </span>
                        {author.institution && <p>{author.institution}</p>}
                    </div>
                ))
            }
        </div>
    )
}