import { describe, expect, it } from "vitest";
import { PRODUCT_HTML, PRODUCT_JS } from "../../src/product/product-ui.js";
import { CreateMembershipSchema } from "../../src/product/product-schema.js";

describe("Stage 15 product surface", () => {
  it("wires the product shell only to real platform APIs", () => {
    for (const path of ["/agents", "/runs", "/approvals", "/memories", "/documents", "/retrieval/search", "/connectors", "/memberships", "/audit", "/usage"]) expect(PRODUCT_JS).toContain(path);
    expect(PRODUCT_HTML).toContain("Workspace access"); expect(PRODUCT_HTML).toContain('id="agent-provider"'); expect(PRODUCT_HTML).not.toContain("fake"); expect(PRODUCT_JS).toContain('providers.map'); expect(PRODUCT_JS).not.toContain("placeholder success");
  });
  it("validates bounded unique membership roles", () => {
    expect(CreateMembershipSchema.parse({ identityId: "fb000000-0000-4000-8000-000000000001", roles: ["audit_viewer"] }).identityType).toBe("user");
    expect(() => CreateMembershipSchema.parse({ identityId: "fb000000-0000-4000-8000-000000000001", roles: ["same", "same"] })).toThrow();
  });
});
