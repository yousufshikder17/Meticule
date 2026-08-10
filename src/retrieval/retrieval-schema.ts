import { z } from "zod";

export const DocumentMetadataSchema = z.record(z.string().trim().min(1).max(100), z.union([z.string().max(1000), z.number().finite(), z.boolean(), z.null()]))
  .refine((value) => Object.keys(value).length <= 50, "Document metadata may contain at most 50 keys")
  .refine((value) => JSON.stringify(value).length <= 20_000, "Document metadata is too large");
export const IngestDocumentSchema = z.object({
  title: z.string().trim().min(1).max(500), mediaType: z.enum(["text/plain", "text/markdown", "application/json"]),
  content: z.string().min(1).max(2_000_000), visibility: z.enum(["private", "tenant"]).default("private"),
  sourceUri: z.string().trim().max(2_000).nullable().default(null), metadata: DocumentMetadataSchema.default({}),
});
export const SearchRetrievalSchema = z.object({
  query: z.string().trim().min(1).max(10_000), metadata: DocumentMetadataSchema.default({}),
  maxChunks: z.int().min(1).max(20).default(5), maxTokens: z.int().min(32).max(20_000).default(2_000),
  minimumScore: z.number().min(-1).max(1).default(0.2),
});
export const DeleteDocumentSchema = z.object({ reason: z.string().trim().min(1).max(2_000).default("user_requested_deletion") });
export const RetrievalCitationSchema = z.object({ documentId: z.uuid(), logicalId: z.uuid(), version: z.int().positive(), chunkId: z.uuid(), chunkIndex: z.int().nonnegative(), title: z.string(), sourceUri: z.string().nullable(), contentHash: z.string().regex(/^[0-9a-f]{64}$/) });
export const RetrievalResultSchema = z.object({ content: z.string(), score: z.number(), tokenEstimate: z.int().positive(), citation: RetrievalCitationSchema, metadata: DocumentMetadataSchema });
export const SearchRetrievalOutputSchema = z.object({ results: z.array(RetrievalResultSchema), totalTokens: z.int().nonnegative(), embeddingProvider: z.string(), embeddingModel: z.string() });
export type IngestDocumentInput = z.infer<typeof IngestDocumentSchema>;
export type SearchRetrievalInput = z.infer<typeof SearchRetrievalSchema>;
export type RetrievalResult = z.infer<typeof RetrievalResultSchema>;
