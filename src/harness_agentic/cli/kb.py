"""``harn kb`` -- building the corpus ``kb_search`` answers from.

Without this the ``retrieval`` toolset could never be used: the index is an FTS5
database that has to be built from documents, and nothing built one. A config
setting pointing at a file that no command can create is a setting that only
ever produces "the index does not exist".

Indexing is a separate step from asking, deliberately. It reads the whole corpus
and is the slow part; a bot answering a customer should be doing one query
against a prepared index, not re-reading a directory of documents.
"""

from __future__ import annotations

from pathlib import Path

import typer

from harness_agentic.cli.render import console, err_console
from harness_agentic.constants import harness_home
from harness_agentic.data.kb import SqliteKnowledgeBase

DEFAULT_PATTERN = "**/*.md"


def default_index() -> Path:
    """Where an index lives when the operator has not said otherwise."""
    return harness_home() / "kb" / "index.sqlite"


def register(app: typer.Typer) -> None:
    """Attach the ``kb`` command group."""
    group = typer.Typer(
        help="Build and inspect the knowledge base the agent answers from.",
        no_args_is_help=True,
    )
    app.add_typer(group, name="kb")

    @group.command("index")
    def index(
        directory: Path = typer.Argument(..., help="Directory of documents to index."),
        pattern: str = typer.Option(DEFAULT_PATTERN, "--pattern", help="Which files to read."),
        index_path: Path | None = typer.Option(None, "--index", help="Where to write it."),
    ) -> None:
        """Index a directory of documents, replacing what each one contributed.

        Re-running is safe and is the intended way to refresh: each document
        replaces its own passages. A document *deleted* from the directory is not
        noticed, because nothing here knows it used to exist -- use ``forget``.
        """
        if not directory.is_dir():
            err_console.print(f"[red]not a directory:[/] {directory}")
            raise typer.Exit(code=1)
        target = index_path if index_path is not None else default_index()
        kb = SqliteKnowledgeBase(path=target)
        written = kb.index_directory(directory.expanduser(), pattern=pattern)
        if not written:
            err_console.print(
                f"[yellow]nothing matched {pattern!r} under {directory}[/] -- "
                f"the index was not changed"
            )
            kb.close()
            raise typer.Exit(code=1)
        console.print(f"[green]indexed[/] {written} passage(s) into {target}")
        console.print(f'[dim]set retrieval.index = "{target}" to let the agent search it[/]')
        kb.close()

    @group.command("search")
    def search(
        query: str = typer.Argument(..., help="What to look for."),
        limit: int = typer.Option(5, "--limit", "-n"),
        index_path: Path | None = typer.Option(None, "--index"),
    ) -> None:
        """Search the index yourself.

        The same query the agent's ``kb_search`` runs, so a corpus that answers
        badly can be diagnosed without a model in the loop.
        """
        kb = _open(index_path)
        passages = kb.search(query, limit=limit)
        if not passages:
            console.print("[dim]no passages matched[/]")
            kb.close()
            return
        for passage in passages:
            console.print(passage.render())
            console.print()
        kb.close()

    @group.command("list")
    def list_documents(index_path: Path | None = typer.Option(None, "--index")) -> None:
        """Show which documents are indexed, and how much each contributed."""
        kb = _open(index_path)
        documents = kb.documents()
        if not documents:
            console.print("[dim]the index is empty[/]")
            kb.close()
            return
        for doc_id, count in documents:
            console.print(f"  {doc_id}  [dim]{count} passage(s)[/]")
        console.print(f"[dim]{kb.count()} passage(s) across {len(documents)} document(s)[/]")
        kb.close()

    @group.command("forget")
    def forget(
        doc_id: str = typer.Argument(..., help="A document id from `kb list`."),
        index_path: Path | None = typer.Option(None, "--index"),
    ) -> None:
        """Drop one document's passages.

        Needed because re-indexing a directory cannot notice a file that is gone:
        it reads what is there, and a document nobody mentions stays.
        """
        kb = _open(index_path)
        removed = kb.forget(doc_id)
        kb.close()
        if not removed:
            err_console.print(f"[red]no document {doc_id!r} in the index[/]")
            raise typer.Exit(code=1)
        console.print(f"[green]forgot[/] {doc_id} ({removed} passage(s))")


def _open(index_path: Path | None) -> SqliteKnowledgeBase:
    """Open an existing index, refusing to create one by accident.

    Opening a missing file would make an empty index and every search would then
    report no matches -- which reads as "the corpus does not cover that" rather
    than "you are searching the wrong file".
    """
    target = index_path if index_path is not None else default_index()
    if not target.is_file():
        err_console.print(f"[red]no index at[/] {target} -- run `harn kb index <directory>` first")
        raise typer.Exit(code=1)
    return SqliteKnowledgeBase(path=target)
