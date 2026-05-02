"""add ts_vec auto-update trigger on chunks

The trigger fires BEFORE INSERT and BEFORE UPDATE OF text. It overrides
any explicitly-supplied ts_vec value — that's intentional. ts_vec is a
derived column; callers should never set it.
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
CREATE FUNCTION chunks_ts_vec_trigger() RETURNS trigger AS $$
BEGIN
  NEW.ts_vec := to_tsvector('english', NEW.text);
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER chunks_ts_vec_update
BEFORE INSERT OR UPDATE OF text ON chunks
FOR EACH ROW EXECUTE FUNCTION chunks_ts_vec_trigger();
""")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS chunks_ts_vec_update ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_ts_vec_trigger()")
