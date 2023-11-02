import unittest
import main
from unittest.mock import patch, MagicMock, Mock, call

class TestRetrainingJob(unittest.TestCase):
    
    @patch('main.DatabaseManager')
    @patch('main.ReTraining')
    @patch('main.logging')  # Mock the logging module directly
    def test_retraining_job(self, mock_logging, mock_retraining, mock_db_manager):
        # Mock logging methods
        mock_logging.info = MagicMock()
        mock_logging.error = MagicMock()

        # Mock database manager instance
        db_instance = mock_db_manager.return_value.__enter__.return_value

        db_instance.get_unprocessed_model_versions.return_value = self.mock_unprocessed_model_versions()

        # Set up mock for get_training_data using the mock functions you've created
        mock_existing_training_records = self.mock_existing_training_record()
        mock_promoted_demoted_records = self.mock_promoted_demoted_record()
        db_instance.get_training_data.return_value = {
            'existing_training_records': mock_existing_training_records,
            'promoted_demoted_records': mock_promoted_demoted_records
        }

        # Mock retraining instance
        mock_retraining_instance = mock_retraining.return_value
        mock_retraining_instance.train_models.return_value = {'rfc': 'model', 'lr': 'model'}

        # Run the retraining job
        main.retraining_job()

        # Check the error log if test fails
        try:
            mock_logging.error.assert_not_called()
        except AssertionError as e:
            error_call_args_list = mock_logging.error.call_args_list
            print("Error logging was called. Args list:", error_call_args_list)
            raise e

        # Expected calls to logging.info
        expected_calls = [
            call('----------------------'),
            call('Retraining Job Iteration Started'),
            call('Fetching Unprocessed Model Versions'),
            call('Unprocessed Models Versions found, Starting Retraining'),
            call('Fetching Training Data from DB'),
            call('Initiating Retraining Process'),
            call('Retraining Process Complete'),
            call('Saving Best Models to Database'),
            call('Succesfully Saved Models to Database'),
            call('Retraining Job Iteration Complete.'),
            call('----------------------')
        ]
        
        mock_logging.info.assert_has_calls(expected_calls)

        # Other assertions...

    def mock_unprocessed_model_versions(self):
        # Set up mock for get_unprocessed_model_versions
        model_version_rfc_mock = MagicMock()
        model_version_rfc_mock.ModelID = 1
        model_version_rfc_mock.ModelVersionID = 1
        model_version_lr_mock = MagicMock()
        model_version_lr_mock.ModelID = 2
        model_version_lr_mock.ModelVersionID = 2

        return [model_version_rfc_mock, model_version_lr_mock]


    def mock_existing_training_record(self):
        # Mock an existing training record similar to the structure of your database record
        record = MagicMock()
        record.TrainingDataID = 1
        record.ItemID = None
        record.CleanText = "plant pest disease management aaron palmateer ph dennis bunkley grower must develop efficient production plan incorporate tactic maximize health minimize opportunity outbreak greenhouse pathogens botrytis talus control"
        record.ISOLanguageCode = 'en'
        record.ScientificName = 'BemisiaTabaci'
        record.GeoRss_Point = 1
        record.SourceName = 'Growertalks'
        record.SourceCountry = 'USA'
        record.SourceRegion = 'North America'
        record.SourceSubject = 'Medical'
        record.Label = 0
        
        return record

    def mock_promoted_demoted_record(self):
        # Mock a promoted/demoted record based on the headers provided
        record = MagicMock()
        record.ItemID = "1014"
        record.Title = "First report of mosaic disease of pumpkin caused by tobacco mosaic virus in Jiangsu Province in China"
        record.KeywrodList='china;disease;agricultural;cucumber;tobacco mosaic;tobacco mosaic virus;virus;supported;work;pumpkin;first report;singapore;cmv;antibodies;grateful;providing antibodies;singapore singapore'
        record.ISOLanguageCode = 'en'
        record.ScientificName = 'Tobacco Mosaic Virus'
        record.GeoRss_Point = 0
        record.SourceName = 'Journal of Plant Pathology'
        record.SourceCountry = 'Germany'
        record.SourceRegion = 'European Union'
        record.SourceSubject = 'Medical'
        record.SelectedByUser = 1
        
        return record

    def setup_mock_training_data(self, db_instance):
        # Use the above functions to create mock lists
        existing_training_records_list_mock = [self.mock_existing_training_record()]
        promoted_demoted_records_list_mock = [self.mock_promoted_demoted_record()]

        # Mock result_data structure
        result_data_mock = {
            'promoted_demoted_records': promoted_demoted_records_list_mock,
            'existing_training_records': existing_training_records_list_mock
        }

        # Set the return value to simulate the database call for training data
        db_instance.get_training_data.return_value = result_data_mock

if __name__ == '__main__':
    unittest.main()
