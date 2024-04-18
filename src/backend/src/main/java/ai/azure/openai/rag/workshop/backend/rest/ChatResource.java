package ai.azure.openai.rag.workshop.backend.rest;

@Path("/chat")
public class ChatResource {
    private static final Logger log = LoggerFactory.getLogger(ChatResource.class);

    private static final String SYSTEM_MESSAGE_PROMPT = """
    Assistant helps the Consto Real Estate company customers with support questions regarding terms of service, privacy policy, and questions about support requests.
    Be brief in your answers.
    Answer ONLY with the facts listed in the list of sources below.
    If there isn't enough information below, say you don't know.
    Do not generate answers that don't use the sources below.
    If asking a clarifying question to the user would help, ask the question.
    For tabular information return it as an html table.
    Do not return markdown format.
    If the question is not in English, answer in the language used in the question.
    Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response.
    Use square brackets to reference the source, for example: [info1.txt].
    Don't combine sources, list each source separately, for example: [info1.txt][info2.pdf].
    """;

    @Inject
    EmbeddingModel embeddingModel;

    @Inject
    EmbeddingStore<TextSegment> embeddingStore;

    @Inject
    ChatLanguageModel chatLanguageModel;

    @POST
    @Consumes({ "application/json" })
    @Produces({ "application/json" })
    public ChatResponse chat(ChatRequest chatRequest) {
        // Embed the question (convert the user's question into vectors that represent
        // the meaning)
        String question = chatRequest.messages.get(chatRequest.messages.size() - 1).content;
        log.info("### Embed the question (convert the question into vectors that represent the meaning) using embeddedQuestion model");
        Embedding embeddedQuestion = embeddingModel.embed(question).content();
        log.debug("# Vector length: {}", embeddedQuestion.vector().length);
        
        // Find relevant embeddings from Qdrant based on the user's question
        log.info("### Find relevant embeddings from Qdrant based on the question");
        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(embeddedQuestion, 3);

        // Builds chat history using the relevant embeddings
        log.info("### Builds chat history using the relevant embeddings");
        List<ChatMessage> chatMessages = new ArrayList<>();
        chatMessages.add(SystemMessage.from(SYSTEM_MESSAGE_PROMPT));
        String userMessage = question + "\n\nSources:\n";
        for (EmbeddingMatch<TextSegment> textSegmentEmbeddingMatch : relevant) {
            userMessage += textSegmentEmbeddingMatch.embedded().metadata("filename") + ": " + textSegmentEmbeddingMatch.embedded().text() + "\n";
        }
        chatMessages.add(UserMessage.from(userMessage));

        // Invoke the LLM
        log.info("### Invoke the LLM");
        Response<AiMessage> response = chatLanguageModel.generate(chatMessages);
        // Return the response
        return ChatResponse.fromMessage(response.content().text());
        
    }
}
