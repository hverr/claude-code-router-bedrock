import { LLMProvider, UnifiedChatRequest } from "@/types/llm";
import { Transformer, TransformerContext } from "@/types/transformer";
import { createHmac, createHash } from "crypto";

export class BedrockTransformer implements Transformer {
  name = "bedrock";
  endPoint = "/model/{modelId}/invoke"; // Will be replaced with actual model ID

  private region: string;
  private service = "bedrock-runtime";

  constructor(options?: any) {
    // Get region from options or default to us-east-1
    this.region = options?.region || process.env.AWS_DEFAULT_REGION || process.env.AWS_REGION || "us-east-1";
  }

  async auth(request: any, provider: LLMProvider, context: TransformerContext): Promise<any> {
    // Get the model ID from the request
    const modelId = request.model || context.req.model;
    if (!modelId) {
      throw new Error("Model ID is required for Bedrock requests");
    }

    // Update the endpoint with the actual model ID
    const endPoint = `/model/${modelId}/invoke${request.stream ? "-with-response-stream" : ""}`;

    // Check if using bearer token format (third-party Bedrock providers)
    if (provider.apiKey && provider.apiKey.startsWith('bedrock-api-key-')) {
      return {
        body: request,
        config: {
          headers: {
            "Authorization": `Bearer ${provider.apiKey}`,
            "Content-Type": "application/json"
          },
          baseURL: `https://bedrock-runtime.${this.region}.amazonaws.com`,
          url: endPoint,
        },
      };
    }

    // Otherwise use AWS Signature V4 authentication
    // Get AWS credentials from environment or provider config
    const accessKeyId = process.env.AWS_ACCESS_KEY_ID || (provider as any).aws_access_key_id || provider.apiKey || "";
    const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY || (provider as any).aws_secret_access_key || "";
    const sessionToken = process.env.AWS_SESSION_TOKEN || (provider as any).aws_session_token;

    if (!accessKeyId || !secretAccessKey) {
      throw new Error("AWS credentials not found. Please provide either:\n  1. A bearer token in api_key (format: bedrock-api-key-...)\n  2. AWS credentials: aws_access_key_id and aws_secret_access_key in the provider config\n  3. Environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY");
    }

    // Create the request details for signing
    const requestDetails = {
      method: "POST",
      host: `bedrock-runtime.${this.region}.amazonaws.com`,
      path: endPoint,
      headers: {
        "Content-Type": "application/json",
        "X-Amz-Date": new Date().toISOString().replace(/[:\-]|\.\d{3}/g, ""),
        "X-Amz-Target": `invokeModel${request.stream ? "WithResponseStream" : ""}`,
        host: `bedrock-runtime.${this.region}.amazonaws.com`,
      },
      body: JSON.stringify(request),
    };

    // Add session token if available
    if (sessionToken) {
      requestDetails.headers["X-Amz-Security-Token"] = sessionToken;
    }

    // Sign the request using AWS Signature V4
    const signedRequest = this.signRequest(
      requestDetails,
      accessKeyId,
      secretAccessKey,
      this.region,
      this.service
    );

    return {
      body: request,
      config: {
        headers: signedRequest.headers,
        baseURL: `https://${requestDetails.host}`,
        url: endPoint,
      },
    };
  }

  async transformRequestIn(request: UnifiedChatRequest): Promise<any> {
    // Convert unified request to Bedrock format
    // Bedrock uses the Anthropic message format for Anthropic models
    const bedrockRequest: any = {
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: request.max_tokens || 1024,
      messages: [],
    };

    // Handle temperature if provided
    if (request.temperature !== undefined) {
      bedrockRequest.temperature = request.temperature;
    }

    // Process messages
    let systemMessage = "";
    for (const message of request.messages) {
      if (message.role === "system") {
        // Collect system messages
        if (typeof message.content === "string") {
          systemMessage += message.content + "\n";
        }
      } else if (message.role === "user" || message.role === "assistant") {
        // Process user and assistant messages
        const bedrockMessage: any = {
          role: message.role,
          content: [],
        };

        if (typeof message.content === "string") {
          bedrockMessage.content = message.content;
        } else if (Array.isArray(message.content)) {
          // Process content array (text and images)
          for (const content of message.content) {
            if (content.type === "text") {
              bedrockMessage.content.push({
                type: "text",
                text: content.text,
              });
            } else if (content.type === "image_url") {
              // Handle image content (if it's base64)
              const imageUrl = content.image_url.url;
              if (imageUrl.startsWith("data:")) {
                // Extract base64 data
                const [, data] = imageUrl.split(",");
                const mediaType = imageUrl.split(";")[0].split(":")[1];

                bedrockMessage.content.push({
                  type: "image",
                  source: {
                    type: "base64",
                    media_type: mediaType,
                    data: data,
                  },
                });
              }
            }
          }
        }

        // Add message if it has content
        if (bedrockMessage.content.length > 0 || typeof bedrockMessage.content === "string") {
          bedrockRequest.messages.push(bedrockMessage);
        }
      }
    }

    // Add system message if present
    if (systemMessage.trim()) {
      bedrockRequest.system = systemMessage.trim();
    }

    // Handle tools if present
    if (request.tools && request.tools.length > 0) {
      bedrockRequest.tools = request.tools.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        input_schema: tool.function.parameters,
      }));

      // Handle tool choice
      if (request.tool_choice) {
        if (request.tool_choice === "auto") {
          bedrockRequest.tool_choice = { type: "auto" };
        } else if (request.tool_choice === "any" || request.tool_choice === "required") {
          bedrockRequest.tool_choice = { type: "any" };
        } else if (typeof request.tool_choice === "object" && request.tool_choice.type === "function") {
          bedrockRequest.tool_choice = {
            type: "tool",
            name: request.tool_choice.function.name
          };
        }
      }
    }

    return bedrockRequest;
  }

  async transformResponseOut(response: Response, context: TransformerContext): Promise<Response> {
    // For Bedrock, we mostly pass through the response
    // but we might need to handle streaming responses differently
    return response;
  }

  // AWS Signature V4 implementation
  private signRequest(
    request: any,
    accessKeyId: string,
    secretAccessKey: string,
    region: string,
    service: string
  ): any {
    const dateTime = new Date().toISOString().replace(/[:\-]|\.\d{3}/g, "");
    const date = dateTime.slice(0, 8);

    // Task 1: Create a canonical request
    const canonicalRequest = this.createCanonicalRequest(request);

    // Task 2: Create a string to sign
    const stringToSign = this.createStringToSign(canonicalRequest, dateTime, region, service);

    // Task 3: Calculate the signature
    const signature = this.calculateSignature(stringToSign, secretAccessKey, date, region, service);

    // Task 4: Add the signature to the request headers
    request.headers["Authorization"] = this.buildAuthorizationHeader(
      accessKeyId,
      signature,
      dateTime,
      region,
      service,
      this.getSignedHeaders(request.headers)
    );

    return request;
  }

  private createCanonicalRequest(request: any): string {
    const method = request.method || "POST";
    const canonicalUri = this.uriEncode(request.path || "/");
    const canonicalQueryString = "";

    // Get sorted headers
    const sortedHeaders = this.getSortedHeaders(request.headers);
    const canonicalHeaders = sortedHeaders.map(([key, value]) => `${key}:${String(value).trim()}\n`).join("");
    const signedHeaders = this.getSignedHeaders(request.headers);

    // Hash the payload
    const payloadHash = createHash("sha256").update(request.body || "").digest("hex");

    return `${method}\n${canonicalUri}\n${canonicalQueryString}\n${canonicalHeaders}\n${signedHeaders}\n${payloadHash}`;
  }

  private createStringToSign(canonicalRequest: string, dateTime: string, region: string, service: string): string {
    const algorithm = "AWS4-HMAC-SHA256";
    const credentialScope = `${dateTime.slice(0, 8)}/${region}/${service}/aws4_request`;
    const hashedCanonicalRequest = createHash("sha256").update(canonicalRequest).digest("hex");

    return `${algorithm}\n${dateTime}\n${credentialScope}\n${hashedCanonicalRequest}`;
  }

  private calculateSignature(
    stringToSign: string,
    secretAccessKey: string,
    date: string,
    region: string,
    service: string
  ): string {
    const kDate = this.hmac(`AWS4${secretAccessKey}`, date);
    const kRegion = this.hmac(kDate, region);
    const kService = this.hmac(kRegion, service);
    const kSigning = this.hmac(kService, "aws4_request");

    return this.hmac(kSigning, stringToSign, "hex");
  }

  private buildAuthorizationHeader(
    accessKeyId: string,
    signature: string,
    dateTime: string,
    region: string,
    service: string,
    signedHeaders: string
  ): string {
    const algorithm = "AWS4-HMAC-SHA256";
    const credentialScope = `${dateTime.slice(0, 8)}/${region}/${service}/aws4_request`;

    return `${algorithm} Credential=${accessKeyId}/${credentialScope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;
  }

  private getSortedHeaders(headers: Record<string, any>): [string, any][] {
    return Object.entries(headers)
      .map(([key, value]) => [key.toLowerCase(), value])
      .sort(([a], [b]) => a.localeCompare(b));
  }

  private getSignedHeaders(headers: Record<string, any>): string {
    return Object.keys(headers)
      .map(key => key.toLowerCase())
      .sort()
      .join(";");
  }

  private uriEncode(uri: string): string {
    return encodeURIComponent(uri).replace(/[!'()*]/g, c => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
  }

  private hmac(key: string | Buffer, data: string, encoding: "hex" | "buffer" = "buffer"): string | Buffer {
    const hmac = createHmac("sha256", key);
    hmac.update(data);
    return encoding === "hex" ? hmac.digest("hex") : hmac.digest();
  }
}