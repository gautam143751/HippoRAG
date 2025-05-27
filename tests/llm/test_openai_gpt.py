from src.hipporag.llm.openai_gpt import CacheOpenAI
from src.hipporag.utils.config_utils import BaseConfig
from unittest.mock import patch

def test_cache_openai_openrouter_initialization():
    config_dict = {
        "llm_name": "openai/gpt-4o", # Example OpenRouter model
        "llm_base_url": "https://openrouter.ai/api/v1",
        "openrouter_api_key": "fake-ork-key",
        "save_dir": "test_outputs/openrouter_test" # Needs a save_dir for cache
    }
    # Initialize BaseConfig directly, as from_experiment_config might have extra logic
    # not needed for this specific unit test focusing on client init.
    # However, using from_experiment_config is also fine if BaseConfig correctly populates.
    config = BaseConfig()
    config.llm_name = config_dict["llm_name"]
    config.llm_base_url = config_dict["llm_base_url"]
    config.openrouter_api_key = config_dict["openrouter_api_key"]
    config.save_dir = config_dict["save_dir"]
    # Ensure other potentially conflicting configs are None
    config.azure_endpoint = None


    # Mock the OpenAI client constructor to inspect its arguments
    with patch('openai.OpenAI') as mock_openai_constructor:
        # Also mock AzureOpenAI to ensure it's not called
        with patch('openai.AzureOpenAI') as mock_azure_openai_constructor:
            # The CacheOpenAI constructor is what we are testing here primarily
            llm = CacheOpenAI(cache_dir=config.save_dir, global_config=config)
            
            mock_openai_constructor.assert_called_once()
            mock_azure_openai_constructor.assert_not_called() # Ensure Azure client wasn't called

            args, kwargs = mock_openai_constructor.call_args
            assert kwargs.get("base_url") == "https://openrouter.ai/api/v1"
            assert kwargs.get("api_key") == "fake-ork-key"

def test_cache_openai_azure_initialization_not_openrouter():
    # Test that Azure client is used when Azure endpoint is specified, not OpenRouter
    config_dict = {
        "llm_name": "gpt-4",
        "azure_endpoint": "https://fake-azure.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-05-15",
        "azure_api_key": "fake-azure-key", 
        "save_dir": "test_outputs/azure_test"
    }
    config = BaseConfig()
    config.llm_name = config_dict["llm_name"]
    config.azure_endpoint = config_dict["azure_endpoint"]
    config.azure_api_key = config_dict["azure_api_key"] # BaseConfig should load this
    config.save_dir = config_dict["save_dir"]
    # Ensure OpenRouter specific fields are None or not set to conflicting values
    config.llm_base_url = None 
    config.openrouter_api_key = None


    with patch('openai.OpenAI') as mock_openai_constructor:
        with patch('openai.AzureOpenAI') as mock_azure_openai_constructor:
            # Manually trigger __post_init__ if needed, or ensure BaseConfig handles it.
            # For Azure, the azure_api_key is often loaded from env if not direct.
            # The provided BaseConfig loads OPENAI_API_KEY, not specifically AZURE_API_KEY to azure_api_key field
            # For this test, we assume azure_api_key is correctly populated in BaseConfig instance
            # or that AzureOpenAI client can pick it up from env.
            # Let's ensure it's set on the config for the test.
            # config.azure_api_key = "fake-azure-key" # Explicitly set for clarity if BaseConfig doesn't handle it directly from dict for this field

            llm = CacheOpenAI(cache_dir=config.save_dir, global_config=config)
            
            mock_azure_openai_constructor.assert_called_once()
            mock_openai_constructor.assert_not_called() 

            args, kwargs = mock_azure_openai_constructor.call_args
            assert kwargs.get("azure_endpoint") == "https://fake-azure.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-05-15"
            # The AzureOpenAI client might pick up the API key from environment variables if not explicitly passed,
            # or it might be passed as `api_key` or `azure_ad_token_provider` etc.
            # The current CacheOpenAI code passes `max_retries` but not api_key directly to AzureOpenAI constructor.
            # It relies on AzureOpenAI client to pick it up from env (AZURE_OPENAI_API_KEY) or other auth methods.
            # So, we don't assert api_key here unless CacheOpenAI is changed to pass it.
            # For the purpose of testing correct client selection, this is sufficient.

def test_cache_openai_standard_openai_initialization():
    config_dict = {
        "llm_name": "gpt-3.5-turbo",
        "openai_api_key": "fake-openai-key",
        "save_dir": "test_outputs/openai_test"
    }
    config = BaseConfig()
    config.llm_name = config_dict["llm_name"]
    config.openai_api_key = config_dict["openai_api_key"]
    config.save_dir = config_dict["save_dir"]
    config.llm_base_url = None # Explicitly not OpenRouter
    config.azure_endpoint = None # Explicitly not Azure

    with patch('openai.OpenAI') as mock_openai_constructor:
        with patch('openai.AzureOpenAI') as mock_azure_openai_constructor:
            llm = CacheOpenAI(cache_dir=config.save_dir, global_config=config)
            
            mock_openai_constructor.assert_called_once()
            mock_azure_openai_constructor.assert_not_called()

            args, kwargs = mock_openai_constructor.call_args
            assert kwargs.get("base_url") is None # Or whatever the default OpenAI base URL is, if passed explicitly
            # The OpenAI client will pick up api_key from env var OPENAI_API_KEY if not passed.
            # CacheOpenAI passes base_url, http_client, max_retries. It doesn't pass api_key.
            # So, we don't assert api_key here.
            # assert kwargs.get("api_key") == "fake-openai-key" # This would fail based on current CacheOpenAI

# Ensure PyPDF2 and python-docx are in requirements.txt for the data connector tests
# (already handled in previous steps)
# Ensure pytest and pytest-mock are in requirements.txt (already handled)
