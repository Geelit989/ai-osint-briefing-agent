import { Workspace } from "@/components/workspace";

export default async function Home({ searchParams }: { searchParams: Promise<{ run?: string | string[] }> }) {
  const params = await searchParams;
  return <Workspace initialRunId={typeof params.run === "string" ? params.run : null} />;
}
