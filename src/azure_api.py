import os
from loguru import logger
from openai import AzureOpenAI
import yaml
import random


credentials = yaml.safe_load(open("credentials.yml"))


def azure_openai_client(
    model,
    api_key=None,
    endpoint=None,
    api_version="2025-01-01-preview"
):
    """
    Create an Azure OpenAI client.

    Args:
        model: Model name in format 'azure-{deployment-name}' (e.g., 'azure-gpt-4o-mini')
        api_key: Azure subscription key (optional, reads from credentials.yml if None)
        endpoint: Azure OpenAI endpoint URL (optional, reads from credentials.yml if None)
        api_version: API version (optional, reads from credentials.yml if None)

    Returns:
        AzureOpenAI client instance
    """
    if api_key is None:
        assert model in credentials, f"Model {model} not found in credentials"

        # Randomly select an API key if multiple are provided
        if "round-robin" in credentials[model]:
            num_keys = len(credentials[model]["round-robin"])
            rand_idx = random.randint(0, num_keys - 1)
            credential = credentials[model]["round-robin"][rand_idx]
        else:
            credential = credentials[model]

        api_key = credential["api_key"]
        endpoint = credential["endpoint"]
        api_version = credential.get("api_version", "2025-01-01-preview")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )

    logger.debug(
        f"Azure OpenAI - API key: ****{api_key[-4:]}, endpoint: {endpoint}, api_version: {api_version}"
    )

    return client


def send_azure_openai_request(
    openai_request,
    model,
    api_key=None,
    endpoint=None,
    api_version="2025-01-01-preview"
):
    """
    Send a request to Azure OpenAI.

    Args:
        openai_request: Request dict in OpenAI format (messages, temperature, etc.)
        model: Model name in format 'azure-{deployment-name}' (e.g., 'azure-gpt-4o-mini')
        api_key: Azure subscription key (optional)
        endpoint: Azure OpenAI endpoint URL (optional)
        api_version: API version (optional)

    Returns:
        Response text from the model
    """
    client = azure_openai_client(
        model,
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version
    )

    # Get deployment name from credentials
    assert model in credentials, f"Model {model} not found in credentials"
    credential = credentials[model]

    if "round-robin" in credential:
        # For round-robin, get the deployment from the first credential
        # In a more sophisticated implementation, you might want to track
        # which credential was selected in azure_openai_client
        num_keys = len(credential["round-robin"])
        rand_idx = random.randint(0, num_keys - 1)
        deployment_name = credential["round-robin"][rand_idx]["deployment"]
    else:
        deployment_name = credential["deployment"]

    logger.debug(f"Using Azure deployment: {deployment_name}")

    response = client.chat.completions.create(
        model=deployment_name,
        **openai_request
    )

    return response.choices[0].message.content
