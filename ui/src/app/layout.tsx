import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SpiceXplorer NEWCAS Demo",
  description: "Conference demo UI for the SpiceXplorer cascode OTA case study"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
