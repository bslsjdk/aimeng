import unittest
from scripts.prepare_speaking_corpus import normalize, record_to_text


class SpeakingCorpusTests(unittest.TestCase):
    def test_belle_record_becomes_dialogue(self):
        text = record_to_text({
            "instruction": "解释什么是神经元。",
            "input": "",
            "output": "神经元是神经系统中的基本信息处理单元。"
        })
        self.assertEqual(text, "用户：解释什么是神经元。\n助手：神经元是神经系统中的基本信息处理单元。\n\n")

    def test_empty_or_malformed_record_is_rejected(self):
        self.assertIsNone(record_to_text({"instruction": "问题", "output": ""}))
        self.assertIsNone(record_to_text({"other": "value"}))

    def test_multiturn_messages_are_supported(self):
        text = record_to_text({"messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"}
        ]})
        self.assertIn("用户：你好", text)
        self.assertIn("助手：你好！", text)

    def test_normalize_removes_excess_whitespace(self):
        self.assertEqual(normalize("  你好  \r\n\r\n\r\n 世界  "), "你好\n\n 世界")


if __name__ == "__main__":
    unittest.main()
