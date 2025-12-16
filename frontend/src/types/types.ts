export type Frame = {
  timestamp: number;
  url: string;
  embedding?: number[];
  scene?: number;
};

export type Scene = {
  id: string | number;
  timestamp: number;
  duration?: number;
  end_timestamp?: number;
  caption?: string;
  request_id?: string;
  url: string;
}
