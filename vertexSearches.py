from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

# (developer): Uncomment these variables before running the sample.
project_id = "kiran-cailin"
location = "global"          # Values: "global", "us", "eu"
engine_id = "upc-codes_1747428051210"
search_query = """[
  {
    "Product Name": "Support Relief Compression Socks",
    "Brand": "Unclear",
    "Product Category": "Health & Personal Care",
    "Package Size/Volume": "Unclear",
    "Quantity": "Many",
    "Price": "Unclear",
    "Promotional Details": "Unclear",
    "Distinguishing Features": "Socks with packaging showing legs."
  },"""
#"do you have ant toilet paper?"


def search_sample(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str,
) -> discoveryengine.services.search_service.pagers.SearchPager:
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    # Optional - only supported for unstructured data: Configuration options for search.
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
            model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble="YOUR_CUSTOM_PROMPT"
            ),
            model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                version="stable",
            ),
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=10,
        content_search_spec=content_search_spec,
        # query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
        #     condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        # ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
        # Optional: Use fine-tuned model for this request
        # custom_fine_tuning_spec=discoveryengine.CustomFineTuningSpec(
        #     enable_search_adaptor=True
        # ),
    )

    page_result = client.search(request)

    # Handle the response
    for response in page_result:
        print(response)

    return page_result

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

def answer_query_sample(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str
) -> discoveryengine.AnswerQueryResponse:
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.ConversationalSearchServiceClient(
        client_options=client_options
    )

    # The full resource name of the Search serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_serving_config"

    # Optional: Options for query phase
    # The `query_understanding_spec` below includes all available query phase options.
    # For more details, refer to https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/QueryUnderstandingSpec
    query_understanding_spec = discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec(
        query_rephraser_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec(
            disable=False,  # Optional: Disable query rephraser
            max_rephrase_steps=1,  # Optional: Number of rephrase steps
        ),
        # Optional: Classify query types
        query_classification_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec(
            types=[
                discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.ADVERSARIAL_QUERY,
                discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.NON_ANSWER_SEEKING_QUERY,
            ]  # Options: ADVERSARIAL_QUERY, NON_ANSWER_SEEKING_QUERY or both
        ),
    )

    # Optional: Options for answer phase
    # The `answer_generation_spec` below includes all available query phase options.
    # For more details, refer to https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/AnswerGenerationSpec
    answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
        ignore_adversarial_query=False,  # Optional: Ignore adversarial query
        ignore_non_answer_seeking_query=False,  # Optional: Ignore non-answer seeking query
        ignore_low_relevant_content=False,  # Optional: Return fallback answer when content is not relevant
        model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
            model_version="gemini-2.0-flash-001/answer_gen/v1",  # Optional: Model to use for answer generation
        ),
        prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
            preamble="Give a detailed answer.",  # Optional: Natural language instructions for customizing the answer.
        ),
        include_citations=True,  # Optional: Include citations in the response
        answer_language_code="en",  # Optional: Language code of the answer
    )

    # Initialize request argument(s)
    request = discoveryengine.AnswerQueryRequest(
        serving_config=serving_config,
        query=discoveryengine.Query(text=search_query),
        session=None,  # Optional: include previous session ID to continue a conversation
        query_understanding_spec=query_understanding_spec,
        answer_generation_spec=answer_generation_spec,
    )

    # Make the request
    response = client.answer_query(request)

    # Handle the response
    print(response)

    return response


if __name__ == "__main__":
    print(search_query)
    #search_sample(project_id, location, engine_id, search_query)
    oo = answer_query_sample(project_id, location, engine_id, search_query) 
    print(type(oo))
    print(dir(oo))
    
    print(oo.answer)
    print(type(oo.answer))
    print(ll)

    print("ok")