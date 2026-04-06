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

"""Tests for provider backend metadata."""

from absl.testing import absltest

from langextract import exceptions
from langextract.providers import backends


class ProviderBackendsTest(absltest.TestCase):

  def test_normalize_provider_selection(self):
    self.assertEqual(
        backends.normalize_provider_selection("GeminiLanguageModel"),
        backends.ProviderFamily.GEMINI,
    )
    self.assertEqual(
        backends.normalize_provider_selection("openai"),
        backends.ProviderFamily.OPENAI,
    )

  def test_match_provider_backend_prefers_explicit_provider(self):
    backend = backends.match_provider_backend(
        model_id="gemini-2.5-flash",
        provider="openai",
    )
    self.assertIsNotNone(backend)
    self.assertEqual(backend.family, backends.ProviderFamily.OPENAI)

  def test_builtin_metadata_exposes_current_defaults(self):
    gemini_backend = backends.get_provider_backend(
        backends.ProviderFamily.GEMINI
    )
    openai_backend = backends.get_provider_backend(
        backends.ProviderFamily.OPENAI
    )

    self.assertIsNotNone(gemini_backend)
    self.assertEqual(
        gemini_backend.default_model_id,
        backends.DEFAULT_GEMINI_MODEL_ID,
    )
    self.assertFalse(gemini_backend.optional_dependency)

    self.assertIsNotNone(openai_backend)
    self.assertTrue(openai_backend.optional_dependency)
    self.assertIsNone(openai_backend.load_schema_class())

  def test_gemini_backend_validates_explicit_model_ids(self):
    gemini_backend = backends.get_provider_backend(
        backends.ProviderFamily.GEMINI
    )

    self.assertIsNotNone(gemini_backend)
    self.assertEqual(
        gemini_backend.resolve_model_id("gemini-2.5-flash"),
        "gemini-2.5-flash",
    )

    with self.assertRaises(exceptions.InferenceConfigError):
      gemini_backend.resolve_model_id("gpt-4o")


if __name__ == "__main__":
  absltest.main()
