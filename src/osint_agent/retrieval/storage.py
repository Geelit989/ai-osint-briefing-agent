import sqlite3


### Create SQLite datebase

db_path = "osint_sys.db"

def create_db():

    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        cur.execute("PRAGMA foreign_keys = ON;")


        cur.execute("""
            create table if not exists documents (
                doc_id text primary key,
                title text not null,
                source text not null,
                published_date text not null,
                url text not null,
                raw_text text not null,
                cleaned_text text not null,
                meta_data text 
            )
        """)

        cur.execute("""
            create table if not exists entities (
                entity_id integer primary key autoincrement,
                ent_text text not null,
                start_char integer not null,
                end_char integer not null,
                label text not null,
                doc_id text not null,
                foreign key (doc_id) references documents (doc_id)

            )
        """)

        con.commit()
        con.close()

        print("Database created successfully.")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    create_db()

