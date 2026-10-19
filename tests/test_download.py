"""Downloader-only regression checks: mocked network, no training."""
import io
import json
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from mla.data import download, source_commit


class DownloadTests(unittest.TestCase):
    def test_remote_underscore_local_hyphen(self):
        commit = 'a' * 40
        urls = []
        def open_url(url, **kwargs):
            urls.append(url)
            return io.BytesIO(b'dataset-content')
        with tempfile.TemporaryDirectory() as folder:
            with patch('mla.data.source_commit', return_value=commit), \
                 patch('mla.data.urllib.request.urlopen', side_effect=open_url):
                download(folder, ['roman-empire'])
            self.assertTrue(urls[0].endswith('/data/roman_empire.npz'))
            path = Path(folder) / 'roman-empire.npz'
            self.assertEqual(path.read_bytes(), b'dataset-content')
            metadata = json.loads(path.with_suffix('.source.json').read_text())
            self.assertEqual(metadata['commit'], commit)
            self.assertEqual(len(metadata['sha256']), 64)

    def test_existing_file_requires_no_network(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'minesweeper.npz').write_bytes(b'existing')
            with patch('mla.data.source_commit') as resolve, \
                 patch('mla.data.urllib.request.urlopen') as network:
                download(folder, ['minesweeper'])
            resolve.assert_not_called()
            network.assert_not_called()

    def test_api_rate_limit_uses_git(self):
        commit = 'b' * 40
        error = urllib.error.HTTPError('https://api.github.com', 403, 'rate limited', {}, None)
        result = subprocess.CompletedProcess([], 0, stdout=commit+'\trefs/heads/main\n')
        with patch('mla.data.urllib.request.urlopen', side_effect=error), \
             patch('mla.data.subprocess.run', return_value=result):
            self.assertEqual(source_commit(), commit)


if __name__ == '__main__':
    unittest.main()
