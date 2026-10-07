from app.models import Block, Document


def make_doc(*settings: str) -> Document:
    blocks = [Block(kind="header", text="BEGIN config-v1")]
    blocks += [Block(kind="setting", text=s) for s in settings]
    blocks.append(Block(kind="footer", text="END config-v1"))
    return Document(blocks=blocks)


def test_valid_document_passes_syntax():
    assert make_doc("a=1", "b=2").validate_syntax() == []


def test_missing_header_rejected():
    doc = Document(blocks=[Block(kind="setting", text="a=1"),
                           Block(kind="footer", text="END")])
    errors = doc.validate_syntax()
    assert any("header" in e for e in errors)


def test_bad_setting_line_rejected():
    doc = make_doc("not-a-setting==")
    assert any("key=value" in e for e in doc.validate_syntax())


def test_only_settings_are_removable():
    doc = make_doc("a=1", "b=2", "c=3")
    assert doc.removable_indexes() == [1, 2, 3]


def test_removal_keeps_syntax_valid():
    doc = make_doc("a=1", "b=2")
    for i in doc.removable_indexes():
        assert doc.without(i).validate_syntax() == []


def test_sensitive_values_masked():
    doc = make_doc("password=hunter2", "mode=strict")
    summary = doc.masked_summary()
    assert "hunter2" not in summary
    assert "password=***" in summary
    assert "mode=strict" in summary


def test_text_roundtrip():
    doc = make_doc("a=1", "b=2")
    again = Document.from_text(doc.to_text())
    assert again.canonical() == doc.canonical()
