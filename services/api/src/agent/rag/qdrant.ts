import { VectorStore, Document, QueryOptions } from "./types";
import { QdrantClient, Schemas } from "@qdrant/js-client-rest";
import { embedModel } from "../model";
import { TextNode } from "llamaindex";

export class QdrantVectorStore implements VectorStore {
  private readonly client: QdrantClient;
  private readonly collectionName: string;
  private readonly vectorName?: string;

  constructor(
    collectionName: string,
    url: string,
    apiKey?: string,
    vectorName?: string,
  ) {
    this.client = new QdrantClient({
      url: url,
      apiKey: apiKey,
    });
    this.collectionName = collectionName;
    this.vectorName = vectorName;
  }

  async query({
    query,
    topK = 5,
    metadata = {},
  }: QueryOptions): Promise<Document[]> {
    const vector = await embedModel.getTextEmbedding(query);

    const points = (
      await this.client.query(this.collectionName, {
        query: vector,
        filter: metadata,
        limit: topK,
        with_vector: true,
        with_payload: true,
      })
    ).points;

    return pointsToDocuments(points, this.vectorName);
  }

  async deleteIndex(): Promise<void> {
    await this.client.deleteCollection(this.collectionName);
  }
}

function pointsToDocuments(
  points: Schemas["ScoredPoint"][],
  vectorName?: string,
) {
  const documents = [];
  for (const point of points) {
    const { metadata = {}, ...data } = parseMetadata(point.payload);
    let vector = null;

    if (
      point.vector &&
      typeof point.vector === "object" &&
      !Array.isArray(point.vector)
    ) {
      // Qdrant supports multiple vectors per entry
      // https://qdrant.tech/documentation/concepts/vectors/#named-vectors
      // @ts-expect-error We ignore other formats like sparse vectors
      vector = point.vector[vectorName];
    } else if (Array.isArray(point.vector)) {
      vector = point.vector as number[];
    }

    const document: Document = {
      id: point.id.toString(),
      embedding: vector,
      score: point.score,
      text: data.text || "",
      metadata,
    };

    documents.push(document);
  }

  return documents;
}

function parseMetadata(
  payload?: Record<string, unknown> | null,
): Partial<TextNode> {
  if (!payload) {
    throw new Error("payload is undefined.");
  }

  const nodeContent = payload["_node_content"];
  if (!nodeContent) {
    throw new Error("_node_content key is not found in the payload");
  }

  if (typeof nodeContent !== "string") {
    throw new Error("_node_content value is not a string.");
  }

  return JSON.parse(nodeContent);
}
