import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';

interface ScanuSelectProps {
  label?: string;
  value: string;
  onValueChange: (value: string) => void;
  options: string[];
  placeholder?: string;
  disabled?: boolean;
}

export function ScanuSelect({
  label,
  value,
  onValueChange,
  options,
  placeholder = 'Select…',
  disabled,
}: ScanuSelectProps) {
  return (
    <div>
      {label ? <label className="block text-sm text-white/70 mb-1.5">{label}</label> : null}
      <Select value={value || undefined} onValueChange={onValueChange} disabled={disabled}>
        <SelectTrigger className="w-full h-10 bg-white/10 border-white/20 text-white hover:bg-white/15 data-[placeholder]:text-white/50">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent className="bg-slate-900 border-white/20 text-white z-[100]">
          {options.map((opt) => (
            <SelectItem
              key={opt}
              value={opt}
              className="text-white focus:bg-white/10 focus:text-white cursor-pointer"
            >
              {opt}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
