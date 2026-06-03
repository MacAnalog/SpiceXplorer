import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { existsSync } from "fs";
import path from "path";

export async function GET(req: NextRequest) {
  const filePath = req.nextUrl.searchParams.get("path");
  if (!filePath) {
    return new NextResponse("Missing path parameter", { status: 400 });
  }

  const resolved = path.resolve(filePath);

  if (!existsSync(resolved)) {
    return new NextResponse("File not found", { status: 404 });
  }

  const content = await readFile(resolved, "utf-8");
  return new NextResponse(content, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
