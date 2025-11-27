export default function CodeBlock({ children }: { children: React.ReactNode }) {
    return (
        <div className="px-6 w-full">
            <pre
                className="relative flex flex-col whitespace-pre overflow-auto p-4 rounded-lg w-full github-light github-dark bg-zinc-200 dark:bg-zinc-800">
                {children}
            </pre>
        </div>
    )
}