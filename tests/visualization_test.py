# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for langextract.visualization."""

import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest

from langextract import data_lib
from langextract import visualization
from langextract.core import data

_PALETTE = visualization._PALETTE
_VISUALIZATION_CSS = visualization._VISUALIZATION_CSS


class VisualizationTest(absltest.TestCase):

  def test_assign_colors_basic_assignment(self):

    extractions = [
        data.Extraction(
            extraction_class="CLASS_A",
            extraction_text="text_a",
            char_interval=data.CharInterval(start_pos=0, end_pos=1),
        ),
        data.Extraction(
            extraction_class="CLASS_B",
            extraction_text="text_b",
            char_interval=data.CharInterval(start_pos=1, end_pos=2),
        ),
    ]
    # Classes are sorted alphabetically before color assignment.
    expected_color_map = {
        "CLASS_A": _PALETTE[0],
        "CLASS_B": _PALETTE[1],
    }

    actual_color_map = visualization._assign_colors(extractions)

    self.assertDictEqual(actual_color_map, expected_color_map)

  def test_build_highlighted_text_single_span_correct_html(self):

    text = "Hello world"
    extraction = data.Extraction(
        extraction_class="GREETING",
        extraction_text="Hello",
        char_interval=data.CharInterval(start_pos=0, end_pos=5),
    )
    extractions = [extraction]
    color_map = {"GREETING": "#ff0000"}
    expected_html = (
        '<span class="lx-highlight lx-current-highlight" data-idx="0" '
        'style="background-color:#ff0000;">Hello</span> world'
    )

    actual_html = visualization._build_highlighted_text(
        text, extractions, color_map
    )

    self.assertEqual(actual_html, expected_html)

  def test_build_highlighted_text_escapes_html_in_text_and_tooltip(self):

    text = "Text with <unsafe> content & ampersand."
    extraction = data.Extraction(
        extraction_class="UNSAFE_CLASS",
        extraction_text="<unsafe> content & ampersand.",
        char_interval=data.CharInterval(start_pos=10, end_pos=39),
        attributes={"detail": "Attribute with <tag> & 'quote'"},
    )
    # Highlighting "<unsafe> content & ampersand"
    extractions = [extraction]
    color_map = {"UNSAFE_CLASS": "#00ff00"}
    expected_highlighted_segment = "&lt;unsafe&gt; content &amp; ampersand."
    expected_html = (
        'Text with <span class="lx-highlight lx-current-highlight"'
        ' data-idx="0" '
        f'style="background-color:#00ff00;">{expected_highlighted_segment}</span>'
    )

    actual_html = visualization._build_highlighted_text(
        text, extractions, color_map
    )

    self.assertEqual(actual_html, expected_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_basic_document_renders_correctly(self):

    doc = data.AnnotatedDocument(
        text="Patient needs Aspirin.",
        extractions=[
            data.Extraction(
                extraction_class="MEDICATION",
                extraction_text="Aspirin",
                char_interval=data.CharInterval(
                    start_pos=14, end_pos=21
                ),  # "Aspirin"
            )
        ],
    )
    # Predictable color based on sorted class name "MEDICATION"
    med_color = _PALETTE[0]
    body_html = (
        'Patient needs <span class="lx-highlight lx-current-highlight"'
        f' data-idx="0" style="background-color:{med_color};">Aspirin</span>.'
    )
    legend_html = (
        '<div class="lx-legend">Highlights Legend: <span class="lx-label" '
        f'style="background-color:{med_color};">MEDICATION</span></div>'
    )
    css_html = _VISUALIZATION_CSS
    expected_components = [
        css_html,
        "lx-animated-wrapper",
        body_html,
        legend_html,
    ]

    actual_html = visualization.visualize(doc)

    # Verify expected components appear in output
    for component in expected_components:
      self.assertIn(component, actual_html)

  @mock.patch.object(
      visualization, "HTML", new=None
  )  # Ensures visualize returns str
  def test_visualize_no_extractions_renders_text_and_empty_legend(self):

    doc = data.AnnotatedDocument(text="No entities here.", extractions=[])
    body_html = (
        '<div class="lx-animated-wrapper"><p>No valid extractions to'
        " animate.</p></div>"
    )
    css_html = _VISUALIZATION_CSS
    expected_html = css_html + "\n" + body_html

    actual_html = visualization.visualize(doc)

    self.assertEqual(actual_html, expected_html)


class VisualizeJsonlMultiDocTest(absltest.TestCase):
  """Tests for multi-document JSONL visualization."""

  def _make_annotated_doc(self, text, extraction_class, span):
    return data.AnnotatedDocument(
        text=text,
        extractions=[
            data.Extraction(
                extraction_class=extraction_class,
                extraction_text=text[span[0]:span[1]],
                char_interval=data.CharInterval(
                    start_pos=span[0], end_pos=span[1]
                ),
            )
        ],
    )

  def _write_jsonl(self, docs):
    """Write AnnotatedDocuments to a temp JSONL file, return its path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
    )
    for doc in docs:
      tmp.write(json.dumps(data_lib.annotated_document_to_dict(doc)) + "\n")
    tmp.close()
    return tmp.name

  @mock.patch.object(visualization, "HTML", new=None)
  def test_jsonl_all_documents_rendered_by_default(self):
    """When document_index is None, all documents should be rendered."""
    doc_a = self._make_annotated_doc("Hello world", "GREETING", (0, 5))
    doc_b = self._make_annotated_doc("Goodbye world", "FAREWELL", (0, 7))
    path = self._write_jsonl([doc_a, doc_b])
    try:
      html_out = visualization.visualize(path)
      # Both docs should produce scoped IDs (_0 and _1)
      self.assertIn("textWindow_0", html_out)
      self.assertIn("textWindow_1", html_out)
    finally:
      os.unlink(path)

  @mock.patch.object(visualization, "HTML", new=None)
  def test_jsonl_document_index_selects_single(self):
    """document_index=1 should render only the second document."""
    doc_a = self._make_annotated_doc("Hello world", "GREETING", (0, 5))
    doc_b = self._make_annotated_doc("Goodbye world", "FAREWELL", (0, 7))
    path = self._write_jsonl([doc_a, doc_b])
    try:
      html_out = visualization.visualize(path, document_index=1)
      # Single doc — no suffix
      self.assertIn("textWindow", html_out)
      self.assertNotIn("textWindow_0", html_out)
      self.assertIn("Goodbye", html_out)
    finally:
      os.unlink(path)

  @mock.patch.object(visualization, "HTML", new=None)
  def test_jsonl_document_index_out_of_range_raises(self):
    doc = self._make_annotated_doc("Hello world", "GREETING", (0, 5))
    path = self._write_jsonl([doc])
    try:
      with self.assertRaises(IndexError):
        visualization.visualize(path, document_index=5)
    finally:
      os.unlink(path)


if __name__ == "__main__":
  absltest.main()
