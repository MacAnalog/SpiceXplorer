import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SpiceXplorer",
  description: "Interactive UI for circuit optimization with SpiceXplorer"
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
