import unittest
from scripts.db import DatabaseManager

class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        # Setup logging for testing
        class MockLogging:
            def info(self, msg):
                print(f"INFO: {msg}")

            def error(self, msg):
                print(f"ERROR: {msg}")

        self.mock_logging = MockLogging()
        self.db_manager = DatabaseManager(self.mock_logging)

    def test_convert_to_int(self):
        self.assertEqual(self.db_manager.convert_to_int(10.0), 10)
        self.assertEqual(self.db_manager.convert_to_int(None), None)

    def test_connection(self):
        with self.db_manager as manager:
            self.assertIsNotNone(manager.connection)
            self.assertIsNotNone(manager.cursor)

    def tearDown(self):
        # Clean up any resources, if necessary
        pass

if __name__ == "__main__":
    unittest.main()
