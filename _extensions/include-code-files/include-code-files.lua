--[[
MIT License

Copyright (c) 2023 2020 Bruno BEAUFILS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]
--- include-code-files.lua – filter to include code from source files

--- Dedent a line
local function dedent(line, n)
	return line:sub(1, n):gsub(" ", "") .. line:sub(n + 1)
end

--- Filter function for code blocks
local function transclude(cb)
	if cb.attributes.include then
		local content = ""
		local fh = io.open(cb.attributes.include)
		if not fh then
			io.stderr:write("Cannot open file " .. cb.attributes.include .. " | Skipping includes\n")
		else
			local number = 1
			local start = 1

			-- change hyphenated attributes to PascalCase
			for i, pascal in pairs({ "startLine", "endLine" }) do
				local hyphen = pascal:gsub("%u", "-%0"):lower()
				if cb.attributes[hyphen] then
					cb.attributes[pascal] = cb.attributes[hyphen]
					cb.attributes[hyphen] = nil
				end
			end

			if cb.attributes.startLine then
				cb.attributes.startFrom = cb.attributes.startLine
				start = tonumber(cb.attributes.startLine)
			end
			for line in fh:lines("L") do
				if cb.attributes.dedent then
					line = dedent(line, cb.attributes.dedent)
				end
				if number >= start then
					if not cb.attributes.endLine or number <= tonumber(cb.attributes.endLine) then
						content = content .. line
					end
				end
				number = number + 1
			end
			fh:close()
		end
		-- remove key-value pair for used keys
		cb.attributes.include = nil
		cb.attributes.startLine = nil
		cb.attributes.endLine = nil
		cb.attributes.dedent = nil
		-- return final code block
		return pandoc.CodeBlock(content, cb.attr)
	end
end

return {
	{ CodeBlock = transclude },
}
