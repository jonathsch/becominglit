type Author = {
  name: string;
  url?: string;
  institution?: string;
  notes?: string[];
}

type LinkEntry = {
  url: string;
  name: string;
  icon?: any;
}

type Note = {
  symbol: string;
  text: string;
}

export { Author, LinkEntry, Note };